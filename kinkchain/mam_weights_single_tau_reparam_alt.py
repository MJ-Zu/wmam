"""Single-tau bulk -> refine -> readout harness, built from mam_weights.py's
building blocks (imported, not modified).

mam_weights.py's cyclic harness trains a whole triangle of tau targets, many
times over, to see whether a Marginally Absorbing Manifold forms. This module
strips that down to ONE fixed tau and runs the three training phases back to
back on it, so the per-step loss dynamics of each phase can be inspected in
isolation:

  * bulk    -- optax.rmsprop(momentum=0.0), lr cosine-decayed from cfg.lr to
               cfg.lr * cfg.bulk_afactor over cfg.max_steps (cold start).
  * refine  -- plain GD ("strict" gradient descent, no momentum / no adaptive
               scaling), fixed lr = cfg.refine_lr, warm-started from bulk's
               final weights.
  * readout -- plain GD, fixed lr = cfg.readout_lr, warm-started from
               refine's final weights. Unlike bulk/refine, this is a GRID
               sweep, mirroring mam_weights.readout_logged: for every
               tau_read on a grid spanning [tau - 0.3, tau + 0.3], retrain
               from the refine endpoint to tau_read and then back to tau.
               This is the actual MAM reversibility probe -- if a manifold
               has formed, the round-trip should leave the near-tau
               accuracy/weights essentially unchanged for tau_read close to
               tau, with degradation showing up as tau_read moves away.
               (The grid legs use a lightweight fixed-step GD trainer with no
               per-step logging -- make_logging_trainer's per-step weight
               buffer is too expensive to pay 2x per grid point; bulk/refine
               below only run ONCE each, so the full per-step log is cheap.)

Bulk/refine each log loss at every step and report test-set accuracy both
overall and restricted to the "near tau" band |a - tau| < cfg.near_band --
the region right at the classification boundary, where a memory effect would
show up first. Readout reports the same near-tau accuracy, but as a function
of tau_read across the grid (before the "to" leg and after the "back" leg),
plus the round-trip weight displacement.

Run from the project root:
    python -m kinkchain.mam_weights_single_tau                  # tau = cfg.tau_max
    python -m kinkchain.mam_weights_single_tau --tau 0.55 --plot runs/single_tau.png
"""

from __future__ import annotations
from dataclasses import dataclass, asdict

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .mam_weights import _build
from .mam_weights_reparam_alt import make_logging_trainer
from .mam import _rms
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
#    data: str=None
    tau_min: float = 0.5
    tau_max: float = 0.6
    increment: float = 0.02
    near_band: float = 0.12      # |a - tau_max| < near_band = "near boundary"
    # --- bulk phase (RMSProp-no-momentum + cosine-decayed lr) ---
    bulk_optimizer: str = "adam"   # "rmsprop" | "adam" | "gd"
    lr: float = 1e-2                  # bulk base lr
    bulk_afactor: float = 0.3         # cosine decay: lr -> lr*afactor over max_steps
    max_steps: int = 100000              # per-target step cap
    tol: float = 1e-20                # per-target convergence tol (classification) 1e-4
    # --- MAM-approach detection ---
    converge_tol: float = 1e-2        # cycle "converged" if every target's loss < this
    # --- refine phase (plain fixed-lr GD) ---
    refine_lr: float = 3e-3
    refine_max_steps: int = 1
    refine_tol: float = 1e-2
    # --- readout (plain GD, MODERATE lr so labels can be re-fit) ---
    readout_lr: float = 3e-3
    
    readout_max_steps: int = 500
    budget_steps: int = 200
    readout_tol: float = 1e-2
    readout_grad_clip: float = 100.0   # clip readout's plain-GD gradients to [-c,c] (None = off)
    # --- optimization noise (off by default; applied only during bulk) ---
    batch_size: int = 0
    augment: bool = False
    seed: int = 0
    seed_kink: int = 100


def _make_plain_gd(model, knots, scales, mask, Xtr, atr, *, steps, lr, grad_clip=None):
    """Fixed-step full-batch plain GD, no per-step logging (jax.lax.fori_loop
    -- no per-step weight-buffer writes). Used for the readout grid, where we
    only need each leg's endpoint, not its trajectory; make_logging_trainer's
    per-step buffer write is the dominant cost and isn't worth paying twice
    per grid point when there can be dozens of them.

    grad_clip: if set, clips each gradient element to [-grad_clip, grad_clip]
    before the SGD step (via optax.clip -- pytree-safe elementwise clipping,
    same semantics as `jnp.where(abs(g) > c, c*sign(g), g)` applied per
    parameter). Plain SGD has no per-parameter rescaling like RMSProp/Adam,
    so a large `s` on kinked params (reparam_alt's x' = s*|x-c|) amplifies
    their gradient by ~s -- with a fixed lr tuned for unamplified gradients,
    that overshoots and diverges to NaN within a few steps. Clipping caps
    that per-parameter blowup directly, instead of retuning lr for a
    specific s.
    """
    def logits(th, X):
        return model.apply({"params": reparam_alt(th, knots, scales, mask)}, X)

    def full_loss(th, tau):
        lab = (atr >= tau).astype(jnp.float32)
        return optax.sigmoid_binary_cross_entropy(logits(th, Xtr), lab).mean()

    g = jax.grad(full_loss)
    opt = (optax.chain(optax.clip(grad_clip), optax.sgd(lr, momentum=None))
           if grad_clip is not None else optax.sgd(lr, momentum=None))

    @jax.jit
    def train(theta, tau):
        st = opt.init(theta)

        def step(_, c):
            th, s = c
            up, s = opt.update(g(th, tau), s, th)
            return optax.apply_updates(th, up), s

        theta, _ = jax.lax.fori_loop(0, steps, step, (theta, st))
        return theta, full_loss(theta, tau)

    return train


def _per_step_test_loss(model, knots, scales, mask, unravel, theta_traj, Xte, ate, tau,
                        stride=1):
    """Test-set (validation) BCE loss at every `stride`-th logged step of a
    trajectory (stride=1 -> every step).

    theta_traj: (n_steps, n_params) flat array (e.g. bt[:ns+1] from a
    make_logging_trainer call -- one row per logged training step).

    Uses jax.lax.map (sequential -- like a compiled for-loop, one step's
    forward pass at a time, memory reused across iterations), NOT jax.vmap.
    vmap would batch ALL steps into one array of shape (n_steps, n_test, H,
    W, channels) -- for max_steps=10000 that outer dimension alone makes
    even a small CNN's conv activations tens of GB (float64), which is what
    was driving this script out of memory. lax.map is slower (no
    cross-step parallelism) but its memory footprint is independent of
    n_steps.
    """
    theta_traj = jnp.asarray(theta_traj[::stride])
    lab = (jnp.asarray(ate) >= tau).astype(jnp.float32)

    def loss_at(theta_flat):
        th = unravel(theta_flat)
        logits = model.apply({"params": reparam_alt(th, knots, scales, mask)}, Xte)
        return optax.sigmoid_binary_cross_entropy(logits, lab).mean()

    return np.asarray(jax.lax.map(loss_at, theta_traj))


def _accuracy(model, knots, scales, mask,theta, X, a, tau, near_band):
    """Overall and near-tau-band test accuracy at threshold 0.5."""
    logits = model.apply({"params": reparam_alt(theta, knots, scales, mask)}, X)
    #logits = model.apply({"params": theta}, X)
    pred = jax.nn.sigmoid(logits) >= 0.5
    lab = jnp.asarray(a) >= tau
    correct = np.asarray(pred == lab)
    near = np.abs(np.asarray(a) - tau) < near_band
    overall = float(correct.mean())
    near_acc = float(correct[near].mean()) if near.any() else float("nan")
    return overall, near_acc, int(near.sum())


def run_single_tau(cfg=None, tau=None, verbose=True, out_path="runs/wmam_single_tau.npz",
                    grid_half_width=0.3, grid_step=0.05, grid_steps=None):
    """Bulk -> refine -> readout at one fixed tau.

    Bulk/refine each return a per-step loss curve and an end-of-phase
    accuracy readout. Readout sweeps tau_read over a grid centered on tau
    (+/- grid_half_width, default +/-0.3, spaced by grid_step, default 0.05):
    for each tau_read it retrains (fixed-lr GD, grid_steps steps -> cfg.
    budget_steps if unset) from the refine endpoint to tau_read and back to
    tau, and reports the near-tau accuracy and weight displacement of the
    round trip.

    Cost note: each grid point is 2 * grid_steps full-batch GD steps with no
    early exit, on top of bulk+refine. On CPU-only JAX this CNN costs ~10ms/
    step, so e.g. 13 points (grid_step=0.05) * 2 * 500 steps ~= 2-3 minutes;
    the default cfg.increment (0.02, 31 points) or cfg.readout_max_steps
    (2000) would take 20-30x longer -- raise grid_step/lower grid_steps for a
    quick look, or pass finer/larger values once you know you want to wait.
    """
    cfg = cfg or WMAMConfig()
    tau = cfg.tau_max if tau is None else float(tau)
    grid_steps = cfg.budget_steps if grid_steps is None else grid_steps
    B = _build(cfg)
    common = dict(model=B["model"], knots=B["knots"], scales=B["scales"], mask=B["mask"],
                  Xtr=B["Xtr"], atr=B["atr"], Xte=B["Xte"], ate=B["ate"])
    flatf = B["flat"]

    bulk_log, _, _, _ = make_logging_trainer(
        **common, max_steps=cfg.max_steps, tol=cfg.tol,
        opt_name=cfg.bulk_optimizer, lr=cfg.lr, afactor=cfg.bulk_afactor)
    refine_log, _, _, _ = make_logging_trainer(
        **common, max_steps=cfg.refine_max_steps, tol=cfg.refine_tol,
        opt_name="gd", lr=cfg.refine_lr, afactor=1.0)
    readout_leg = _make_plain_gd(B["model"], B["knots"], B["scales"], B["mask"],
                                  B["Xtr"], B["atr"],
                                  steps=grid_steps, lr=cfg.readout_lr,
                                  grad_clip=cfg.readout_grad_clip)

    dummy = jax.random.PRNGKey(0)
    theta = B["params0"]
    phases = {}
    for name, trainer in (("bulk", bulk_log), ("refine", refine_log)):
        theta, ns, l, bt, bl = trainer(theta, tau, dummy)
        loss = np.asarray(bl[: int(ns) + 1])
        theta_traj = np.asarray(bt[: int(ns) + 1])
        test_loss = _per_step_test_loss(B["model"], B["knots"], B["scales"], B["mask"],
                                        B["unravel"], theta_traj, B["Xte"], B["ate"], tau)
        overall, near_acc, n_near = _accuracy(
            B["model"], B["knots"], B["scales"], B["mask"], theta, B["Xte"], B["ate"], tau, cfg.near_band)
        phases[name] = dict(loss=loss, theta_traj=theta_traj, test_loss=test_loss, n_steps=int(ns),
                             final_loss=float(l), final_test_loss=float(test_loss[-1]),
                             test_acc=overall, near_tau_acc=near_acc, n_near=n_near)
        if verbose:
            print(f"[{name:7s}] steps={int(ns):5d}  final_loss={float(l):.3e}  n_thetas={int(len(flatf(theta))):6d}"
                  f"final_test_loss={float(test_loss[-1]):.3e}  "
                  f"test_acc={overall:.3f}  "
                  f"near_tau_acc(|a-{tau:g}|<{cfg.near_band})={near_acc:.3f} (n={n_near})")

    theta_anchor = theta  # refine's endpoint; readout probes reversibility around it
    grid = np.round(np.arange(tau - grid_half_width, tau + grid_half_width + 1e-9, grid_step), 6)

    final_loss_to, final_loss_back = [], []
    near_acc_at_read, near_acc_back, dW = [], [], []
    for tr in grid:
        th_to, l1 = readout_leg(theta_anchor, float(tr))
        final_loss_to.append(float(l1))
        _, na_read, _ = _accuracy(B["model"], B["knots"], B["scales"], B["mask"], th_to,
                                   B["Xte"], B["ate"], float(tr), cfg.near_band)
        near_acc_at_read.append(na_read)

        th_back, l2 = readout_leg(th_to, tau)
        final_loss_back.append(float(l2))
        _, na_back, _ = _accuracy(B["model"], B["knots"], B["scales"], B["mask"], th_back,
                                   B["Xte"], B["ate"], tau, cfg.near_band)
        near_acc_back.append(na_back)
        dW.append(float(_rms(jnp.asarray(flatf(th_back) - flatf(theta_anchor)))))

    phases["readout"] = dict(
        grid=grid,
        final_loss_to=np.array(final_loss_to), final_loss_back=np.array(final_loss_back),
        near_acc_at_read=np.array(near_acc_at_read), near_acc_back=np.array(near_acc_back),
        dW=np.array(dW))
    if verbose:
        j_tau = int(np.argmin(np.abs(grid - tau)))
        print(f"[readout] grid={len(grid)} pts in [{grid[0]:g}, {grid[-1]:g}] step={grid_step:g}  "
              f"near_tau_acc after round-trip: mean={np.mean(near_acc_back):.3f}  "
              f"min={np.min(near_acc_back):.3f}  at-tau={near_acc_back[j_tau]:.3f}  "
              f"dW: mean={np.mean(dW):.2e}  max={np.max(dW):.2e}")

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
            refine_near_tau_acc=phases["refine"]["near_tau_acc"],
            ro_grid=grid, ro_final_loss_to=phases["readout"]["final_loss_to"],
            ro_final_loss_back=phases["readout"]["final_loss_back"],
            ro_near_acc_at_read=phases["readout"]["near_acc_at_read"],
            ro_near_acc_back=phases["readout"]["near_acc_back"], ro_dW=phases["readout"]["dW"])
        if verbose:
            print(f"saved -> {out_path}")
    return phases


def plot_single_tau(phases, tau, save_path=None):
    """Top: bulk/refine loss (log-y) vs step, concatenated with a divider.
    Bottom: readout grid diagnostics vs tau_read -- near-tau accuracy after
    the round trip (left axis) and round-trip weight displacement dW (right
    axis), with a marker at tau_read == tau."""
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 8))

    colors = {"bulk": "C0", "refine": "C1"}
    offset = 0
    for name in ("bulk", "refine"):
        loss = phases[name]["loss"]
        test_loss = phases[name].get("test_loss")
        step = offset + np.arange(len(loss))
        a1.semilogy(step, loss, color=colors[name], lw=1.2, label=f"{name} (train)")
        if test_loss is not None:
            a1.semilogy(step, test_loss, color=colors[name], lw=1.0, ls=":",
                        label=f"{name} (test)")
        offset += len(loss)
        a1.axvline(offset - 0.5, color="0.8", lw=0.8, ls="--")
    a1.set_xlabel("step (bulk | refine, concatenated)")
    a1.set_ylabel("loss (log)")
    a1.set_title(f"single-tau training, tau={tau:g}")
    a1.legend()

    ro = phases["readout"]
    grid = ro["grid"]
    a2.plot(grid, ro["near_acc_back"], color="C2", marker="o", ms=3, lw=1.2,
            label="near-tau accuracy after round-trip")
    a2.axvline(tau, color="0.6", lw=0.8, ls="--")
    a2.set_xlabel("tau_read")
    a2.set_ylabel("near-tau accuracy", color="C2")
    a2.set_ylim(-0.02, 1.02)
    at = a2.twinx()
    at.semilogy(grid, ro["dW"], color="C3", marker="s", ms=3, lw=1.0,
                label="weight displacement dW")
    at.set_ylabel("dW (log)", color="C3")
    a2.set_title("readout: reversibility around tau")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"saved plot -> {save_path}")
    return fig


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Single-tau bulk -> refine -> readout training, logging "
                    "per-step loss and near-tau test accuracy at each phase.")
    p.add_argument("--tau", type=float, default=None,
                   help="fixed tau to train to (default: cfg.tau_max)")
    p.add_argument("--grid-half-width", type=float, default=0.3,
                   help="readout tau_read grid spans [tau-w, tau+w] (default 0.3)")
    p.add_argument("--grid-step", type=float, default=0.05,
                   help="readout grid spacing (default 0.05; cfg.increment=0.02 "
                        "is finer but ~2.5x more grid points)")
    p.add_argument("--grid-steps", type=int, default=None,
                   help="GD steps per readout leg (default: cfg.budget_steps)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seed-kink", type=int, default=None)
    p.add_argument("--kink-fraction", type=float, default=None,
                   help="fraction of parameters that are kinked (default: cfg's own default)")
    p.add_argument("--out", default="runs/wmam_single_tau.npz")
    p.add_argument("--plot", default=None, help="path to save a loss-curve PNG")
    return p.parse_args()


def main():
    a = _parse_args()
    overrides = {}
    if a.kink_fraction is not None:
        overrides["kink_fraction"] = a.kink_fraction
    if a.seed is not None:
        overrides["seed"] = a.seed
    if a.seed_kink is not None:
        overrides["seed_kink"] = a.seed_kink
    cfg = WMAMConfig(**overrides)
    print("kink fraction:{}".format(cfg.kink_fraction))

    phases = run_single_tau(cfg, tau=a.tau, out_path=a.out,
                             grid_half_width=a.grid_half_width, grid_step=a.grid_step,
                             grid_steps=a.grid_steps)
    if a.plot:
        plot_single_tau(phases, a.tau if a.tau is not None else cfg.tau_max, save_path=a.plot)


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    main()
