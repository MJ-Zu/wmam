"""Single-tau bulk -> refine training with config load/save (no readout).

Merges mam_weights_single_tau_reparam_alt.py (run_single_tau) and
mam_weights_single_tau_cfg_io.py (load/save the WMAMConfig as JSON); neither of
those files is modified. Only the two training phases are kept -- everything
that belonged to the readout (the tau_read grid sweep, its plain-GD leg trainer,
the knot-crossing analysis of readout results, the grid CLI flags) is gone.

Runs ONE fixed tau through two phases back to back, logging loss at every step
(or every cfg.log_every-th step: the rows of the saved bulk_theta / bulk_loss / ... are then
steps 0, k, 2k, ... plus the final state, recorded during training so the device buffer
shrinks k-fold; the key log_every is saved in the .npz):

  * bulk   -- cfg.bulk_optimizer ("adam" | "rmsprop" | "gd"), lr cosine-decayed
              from cfg.lr to cfg.lr * cfg.bulk_afactor over cfg.max_steps (cold
              start from the kinked init).
  * refine -- plain GD (no momentum / adaptive scaling), fixed lr = cfg.refine_lr,
              warm-started from bulk's final weights.

Each phase reports the per-step train loss, the per-step test loss, and the
end-of-phase test accuracy overall and in the near-tau band |a - tau| < near_band.

Config: the full resolved WMAMConfig (seed and seed_kink included) is written to
a JSON next to the output (<out>_cfg.json), and a base config can be loaded from
such a file with --load-cfg. CLI overrides are applied ON TOP of the loaded base
with dataclasses.replace, so a flag you do not pass keeps the loaded value.

The readout hyper-parameters (readout_*, budget_steps) stay in WMAMConfig even
though nothing here uses them: the cfg JSON is shared with the readout scripts,
and dropping the fields would make existing JSON files fail to load.

Run from the project root:
    python -m kinkchain.mam_weights_single_tau_train --tau 0.55 --seed 15467 --seed-kink 13900 \
        --out runs/wmam_single_tau0.55_kink0.0_15467_13900.npz
    python -m kinkchain.mam_weights_single_tau_train --load-cfg runs/wmam_single_tau_cfg.json --seed 999
    python -m kinkchain.mam_weights_single_tau_train --load-cfg runs/base_cfg.json --cfg-only
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .mam_weights import _build
from .mam_weights_reparam_alt import make_logging_trainer, _truncate_logged
from .reparam import reparam_alt


@dataclass
class WMAMConfig:
    img: int = 16               # image size 16x16
    n_train: int = 100           # small -> underdetermined (task-hardening)
    n_test: int = 150
    channels: int = 12		# number of channels in CNN
    kink_fraction: float = 0.2
    s_low: float = 50.          # amplitude of kinks s, theta = theta + s|theta-c|
    s_high: float = 50.0
    property_noise: float = 0.0  # noise in the image's encoding of a (hardening)
    data: str = "data/disk_data_100_150.npz"
    tau_min: float = 0.5
    tau_max: float = 0.6
    increment: float = 0.02
    near_band: float = 0.12      # |a - tau_max| < near_band = "near boundary"
    # --- bulk phase (cosine-decayed lr) ---
    bulk_optimizer: str = "adam"   # "rmsprop" | "adam" | "gd"
    bulk_lr: float = 1e-3                  # bulk base lr
    bulk_afactor: float = 0.3         # cosine decay: lr -> lr*afactor over max_steps
    bulk_max_steps: int = 100000              # per-target step cap
    bulk_tol: float = 1e-20                # per-target convergence tol (classification) 1e-4
    converge_tol: float = 1e-2
    # --- refine phase (plain fixed-lr GD) ---
    refine_optimizer: str = "gd"   # "rmsprop" | "adam" | "gd"
    refine_afactor: float = 1.0
    refine_lr: float = 3e-4
    refine_max_steps: int = 1
    refine_tol: float = 1e-20
    # --- readout settings: unused here, kept so cfg JSON files stay interchangeable ---
    readout_lr: float = 3e-4
    readout_max_steps: int = 500
    budget_steps: int = 200
    readout_tol: float = 1e-20
    # --- optimization noise (off by default; applied only during bulk) ---
    batch_size: int = 0
    augment: bool = False
    seed: int = 0
    seed_kink: int = 100
    # --- logging: record theta / loss only every log_every-th step (plus each phase's final
    # state) during training; 1 = every step. Shrinks the (max_steps x n_params) device buffer.
    log_every: int = 1


def load_cfg(path):
    """Read a WMAMConfig back from a JSON file written by save_cfg."""
    with open(path) as f:
        data = json.load(f)
    return WMAMConfig(**data)


def save_cfg(cfg, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
    return path


def _default_cfg_path(out_path):
    root, _ext = os.path.splitext(out_path)
    return root + "_cfg.json"


def _per_step_test_loss(model, knots, scales, mask, unravel, theta_traj, Xte, ate, tau,
                        stride=1):
    """Test-set (validation) BCE loss at every `stride`-th logged step of a
    trajectory (stride=1 -> every step).

    theta_traj: (n_steps, n_params) flat array (e.g. bt[:ns+1] from a
    make_logging_trainer call -- one row per logged training step).

    Uses jax.lax.map (sequential, memory independent of n_steps), NOT jax.vmap:
    vmap would batch ALL steps into one (n_steps, n_test, H, W, channels) array,
    which for max_steps=10000 makes even a small CNN's activations tens of GB.
    """
    theta_traj = jnp.asarray(theta_traj[::stride])
    lab = (jnp.asarray(ate) >= tau).astype(jnp.float32)

    def loss_at(theta_flat):
        th = unravel(theta_flat)
        logits = model.apply({"params": reparam_alt(th, knots, scales, mask)}, Xte)
        return optax.sigmoid_binary_cross_entropy(logits, lab).mean()

    return np.asarray(jax.lax.map(loss_at, theta_traj))


def _accuracy(model, knots, scales, mask, theta, X, a, tau, near_band):
    """Overall and near-tau-band test accuracy at threshold 0.5."""
    logits = model.apply({"params": reparam_alt(theta, knots, scales, mask)}, X)
    pred = jax.nn.sigmoid(logits) >= 0.5
    lab = jnp.asarray(a) >= tau
    correct = np.asarray(pred == lab)
    near = np.abs(np.asarray(a) - tau) < near_band
    overall = float(correct.mean())
    near_acc = float(correct[near].mean()) if near.any() else float("nan")
    return overall, near_acc, int(near.sum())


def run_single_tau(cfg=None, tau=None, verbose=True, out_path="runs/wmam_single_tau.npz",
                   grad_clip=None, setup=None):
    """Bulk -> refine at one fixed tau. Returns {"bulk": {...}, "refine": {...}},
    each with the per-step loss / theta trajectory / test loss, the final losses
    and the end-of-phase test accuracy (overall and near tau).

    grad_clip: if set, clips each gradient element to [-c, c] in both phases.
    setup: a frozen arrays-and-model bundle (setup_io.load_setup); built from cfg if None.
    """
    cfg = cfg or WMAMConfig()
    tau = cfg.tau_max if tau is None else float(tau)
    B = setup if setup is not None else _build(cfg)
    common = dict(model=B["model"], knots=B["knots"], scales=B["scales"], mask=B["mask"],
                  Xtr=B["Xtr"], atr=B["atr"], Xte=B["Xte"], ate=B["ate"])
    flatf = B["flat"]

    bulk_log, _, _, _ = make_logging_trainer(
        **common, max_steps=cfg.bulk_max_steps, tol=cfg.bulk_tol,
        opt_name=cfg.bulk_optimizer, lr=cfg.bulk_lr, afactor=cfg.bulk_afactor,
        grad_clip=grad_clip, log_every=cfg.log_every)
    refine_log, _, _, _ = make_logging_trainer(
        **common, max_steps=cfg.refine_max_steps, tol=cfg.refine_tol,
        opt_name=cfg.refine_optimizer, lr=cfg.refine_lr, afactor=cfg.refine_afactor, 
        grad_clip=grad_clip, log_every=cfg.log_every)

    dummy = jax.random.PRNGKey(0)
    theta = B["params0"]
    phases = {}
    for name, trainer in (("bulk", bulk_log), ("refine", refine_log)):
        theta, ns, l, bt, bl = trainer(theta, tau, dummy)
        theta_traj, loss = _truncate_logged(bt, bl, ns, cfg.log_every)   # rows: steps 0, k, 2k, ..., final
        test_loss = _per_step_test_loss(B["model"], B["knots"], B["scales"], B["mask"],
                                        B["unravel"], theta_traj, B["Xte"], B["ate"], tau)
        overall, near_acc, n_near = _accuracy(
            B["model"], B["knots"], B["scales"], B["mask"], theta, B["Xte"], B["ate"], tau, cfg.near_band)
        phases[name] = dict(loss=loss, theta_traj=theta_traj, test_loss=test_loss, n_steps=int(ns),
                            final_loss=float(l), final_test_loss=float(test_loss[-1]),
                            test_acc=overall, near_tau_acc=near_acc, n_near=n_near)
        if verbose:
            print(f"[{name:7s}] steps={int(ns):5d}  final_loss={float(l):.3e}  n_thetas={int(len(flatf(theta))):6d}  "
                  f"final_test_loss={float(test_loss[-1]):.3e}  "
                  f"test_acc={overall:.3f}  "
                  f"near_tau_acc(|a-{tau:g}|<{cfg.near_band})={near_acc:.3f} (n={n_near})")

    if out_path:
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            out_path, tau=tau, kink=B["kink"], theta0=flatf(B["params0"]),
            bulk_loss=phases["bulk"]["loss"], bulk_theta=phases["bulk"]["theta_traj"],
            refine_loss=phases["refine"]["loss"],
            bulk_test_loss=phases["bulk"]["test_loss"], refine_test_loss=phases["refine"]["test_loss"],
            bulk_test_acc=phases["bulk"]["test_acc"], refine_test_acc=phases["refine"]["test_acc"],
            bulk_near_tau_acc=phases["bulk"]["near_tau_acc"],
            refine_near_tau_acc=phases["refine"]["near_tau_acc"], log_every=cfg.log_every)
        if verbose:
            print(f"saved -> {out_path}")
    return phases


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Single-tau bulk -> refine training, logging per-step loss and "
                    "near-tau test accuracy; the base WMAMConfig can be loaded from "
                    "JSON and the fully-resolved cfg (seed/seed_kink included) is "
                    "saved back out alongside the run's output.")
    p.add_argument("--load-cfg", default=None,
                   help="JSON file (from a prior save_cfg) to use as the base cfg "
                        "instead of WMAMConfig()'s defaults")
    p.add_argument("--save-cfg", default=None,
                   help="where to save the resolved cfg (default: --out with .npz "
                        "replaced by _cfg.json)")
    p.add_argument("--no-save-cfg", action="store_true", help="skip saving the resolved cfg")
    p.add_argument("--cfg-only", action="store_true",
                   help="resolve and save the cfg, print it, then exit -- no training")
    p.add_argument("--tau", type=float, default=None,
                   help="fixed tau to train to (default: cfg.tau_max)")
#   p.add_argument("--optimizer", type=str, default=None,
#                  help="bulk optimizer: adam | rmsprop | gd (default: the cfg's)")
#   p.add_argument("--max-steps", type=int, default=None)
#   p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seed-kink", type=int, default=None)
    p.add_argument("--kink-fraction", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--log-every", type=int, default=None,
                   help="record theta/loss only every N-th step during training (default: the cfg's, 1 = every step)")
    p.add_argument("--scale", type=float, default=None,
                   help="kink amplitude s: sets s_low = s_high (default: the cfg's)")
    p.add_argument("--out", default="runs/wmam_single_tau.npz")
    return p.parse_args()


def main():
    a = _parse_args()

    base = load_cfg(a.load_cfg) if a.load_cfg else WMAMConfig()
    overrides = {}
    if a.kink_fraction is not None:
        overrides["kink_fraction"] = a.kink_fraction
#   if a.max_steps is not None:
#       overrides["max_steps"] = a.max_steps
#   if a.lr is not None:
#       overrides["lr"] = a.lr
    if a.seed is not None:
        overrides["seed"] = a.seed
    if a.seed_kink is not None:
        overrides["seed_kink"] = a.seed_kink
    if a.log_every is not None:
        overrides["log_every"] = a.log_every
    if a.scale is not None:
        overrides["s_low"] = a.scale
        overrides["s_high"] = a.scale
#   if a.optimizer is not None:
#       overrides["bulk_optimizer"] = a.optimizer
    cfg = replace(base, **overrides) if overrides else base

    if not a.no_save_cfg:
        cfg_path = a.save_cfg or _default_cfg_path(a.out)
        save_cfg(cfg, cfg_path)
        print(f"[cfg] saved resolved config (seed={cfg.seed}, seed_kink={cfg.seed_kink}) -> {cfg_path}")

    if a.cfg_only:
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
        return

    run_single_tau(cfg, tau=a.tau, grad_clip=a.grad_clip, out_path=a.out)


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    main()
