"""Single-tau readout anchored on an ARBITRARY bulk step, not just the
refine/bulk endpoint. Built from mam_weights_single_tau_reparam_alt.py's
pieces (that file, and the base mam_weights.py / mam_weights_reparam_alt.py,
are not modified).

run_single_tau (mam_weights_single_tau_reparam_alt.py) always runs bulk ->
refine and does the readout grid sweep anchored on the FULLY TRAINED (refine
endpoint) state. This module instead runs bulk ONCE, picks out theta at a
given LOGGED STEP INDEX within that same bulk trajectory, and runs the
identical readout grid sweep anchored there -- so you can see how the
reversibility profile looks partway through training, not only at the end.

Uses the exact same WMAMConfig, data loading (_build), model, and kink
selection (same seed/seed_kink/kink_fraction plumbing) as
mam_weights_single_tau_reparam_alt.py, so a --step-idx run here is directly
comparable to that script's own output for the SAME underlying setup.

Run from the project root:
    python -m kinkchain.mam_weights_single_tau_readout_step --step-idx 50
    python -m kinkchain.mam_weights_single_tau_readout_step --step-idx 500 --kink-fraction 0.3 --plot runs/step500.png
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from .mam_weights import _build
from .mam_weights_reparam_alt import make_logging_trainer
from .mam_weights_single_tau_reparam_alt import WMAMConfig, _make_plain_gd, _accuracy
from .mam import _rms
from .reparam import reparam_alt


def readout_at_step(cfg=None, tau=None, step_idx=0, verbose=True,
                     out_path="runs/wmam_single_tau_readout_step.npz",
                     grid_half_width=0.3, grid_step=0.05, grid_steps=None,
                     load=None):
    """Anchor a readout grid sweep on theta at `step_idx` of a bulk
    trajectory, either freshly trained or loaded from a saved run.

    step_idx=0 is the untrained initial weights (params0); step_idx==n_bulk
    (the actual number of logged steps, <= cfg.max_steps if bulk exited
    early on cfg.tol) is the state run_single_tau's bulk phase ends on.

    load: path to a .npz saved by mam_weights_single_tau_reparam_alt.run_
    single_tau (must contain 'bulk_theta', 'bulk_loss', 'tau', 'kink'). If
    given, the bulk trajectory is loaded from there instead of retraining --
    `cfg` must still describe the SAME setup that produced the file (same
    img/channels/kink_fraction/seed/seed_kink/data), since knots/scales/
    params0 aren't stored in the file and are regenerated via _build(cfg);
    this is checked by comparing the file's saved kink mask against
    _build(cfg)'s own -- a mismatch raises, rather than silently computing a
    readout against the wrong reparam.

    Returns a dict: anchor diagnostics (anchor_loss/anchor_test_acc/
    anchor_near_tau_acc) plus the readout grid arrays (final_loss_to/back,
    near_acc_at_read, near_acc_back, dW), same shapes/semantics as
    run_single_tau's "readout" phase.
    """
    cfg = cfg or WMAMConfig()
    B = _build(cfg)
    flatf = B["flat"]

    if load:
        z = np.load(load, allow_pickle=False)
        saved_kink = z["kink"]
        if saved_kink.shape != B["kink"].shape or not np.array_equal(saved_kink, B["kink"]):
            raise ValueError(
                f"kink mask loaded from {load} (shape {saved_kink.shape}) does not match "
                f"this cfg's _build(cfg) kink mask (shape {B['kink'].shape}) -- pass the "
                f"--img/--channels/--kink-fraction/--seed/--seed-kink/--data that produced "
                f"this file so knots/scales/params0 are regenerated identically.")
        bt = z["bulk_theta"]
        bl = z["bulk_loss"]
        tau = float(z["tau"]) if tau is None else float(tau)
        ns = bt.shape[0] - 1
        if verbose:
            print(f"[bulk]   loaded {ns} logged steps from {load} (no retraining)")
    else:
        tau = cfg.tau_max if tau is None else float(tau)
        common = dict(model=B["model"], knots=B["knots"], scales=B["scales"], mask=B["mask"],
                      Xtr=B["Xtr"], atr=B["atr"], Xte=B["Xte"], ate=B["ate"])
        bulk_log, _, _, _ = make_logging_trainer(
            **common, max_steps=cfg.max_steps, tol=cfg.tol,
            opt_name=cfg.bulk_optimizer, lr=cfg.lr, afactor=cfg.bulk_afactor)
        dummy = jax.random.PRNGKey(0)
        _theta_final, ns, l, bt, bl = bulk_log(B["params0"], tau, dummy)
        bt, bl, ns = np.asarray(bt), np.asarray(bl), int(ns)
        if verbose:
            print(f"[bulk]   ran {ns} steps this run (final_loss={float(l):.3e})")

    ro_log, _, _, _ = make_logging_trainer(
        model=B["model"], knots=B["knots"], scales=B["scales"], mask=B["mask"],
        Xtr=B["Xtr"], atr=B["atr"], Xte=B["Xte"], ate=B["ate"],
        max_steps=grid_steps if grid_steps is not None else cfg.budget_steps,
        tol=cfg.readout_tol, opt_name="gd", lr=cfg.readout_lr, afactor=1.0,
        grad_clip=cfg.readout_grad_clip)
    dummy = jax.random.PRNGKey(0)

    if not (0 <= step_idx <= ns):
        raise ValueError(f"step_idx={step_idx} out of range: bulk has {ns} "
                          f"logged steps (valid range 0..{ns})")

    theta_anchor = B["unravel"](jnp.asarray(bt[step_idx]))
    weight_anchor = reparam_alt(theta_anchor, B["knots"], B["scales"], B["mask"])
    step_loss = float(bl[step_idx])

    overall, near_acc, n_near = _accuracy(
        B["model"], B["knots"], B["scales"], B["mask"], theta_anchor,
        B["Xte"], B["ate"], tau, cfg.near_band)
    if verbose:
        print(f"[anchor] step_idx={step_idx}/{ns}  loss={step_loss:.3e}  "
              f"test_acc={overall:.3f}  "
              f"near_tau_acc(|a-{tau:g}|<{cfg.near_band})={near_acc:.3f} (n={n_near})")

    grid = np.round(np.arange(tau - grid_half_width, tau + grid_half_width + 1e-9, grid_step), 6)
    final_loss_to, final_loss_back = [], []
    near_acc_at_read, near_acc_back, dtheta, dW = [], [], [], []
    # per-step recording (loss, theta, tau), similar to mam_weights_reparam_alt.py's
    # readout_logged: one row per logged step of every "to"/"back" leg.
    rec_cycle, rec_tau, rec_leg, rec_loss, rec_theta = [], [], [], [], []

    def rec(idx, tr, leg, ns_leg, bt_leg, bl_leg):
        k = int(ns_leg) + 1
        rec_cycle.extend([idx] * k); rec_tau.extend([tr] * k)
        rec_leg.extend([leg] * k); rec_loss.extend(np.asarray(bl_leg[:k]).tolist())
        rec_theta.append(np.asarray(bt_leg[:k]))

    for idx, tr in enumerate(grid):
        th_to, ns1, l1, bt1, bl1 = ro_log(theta_anchor, float(tr), dummy)
        rec(idx, float(tr), "to", ns1, bt1, bl1)
        final_loss_to.append(float(l1))
        _, na_read, _ = _accuracy(B["model"], B["knots"], B["scales"], B["mask"], th_to,
                                   B["Xte"], B["ate"], float(tr), cfg.near_band)
        near_acc_at_read.append(na_read)

        th_back, ns2, l2, bt2, bl2 = ro_log(th_to, tau, dummy)
        rec(idx, tau, "back", ns2, bt2, bl2)
        final_loss_back.append(float(l2))
        _, na_back, _ = _accuracy(B["model"], B["knots"], B["scales"], B["mask"], th_back,
                                   B["Xte"], B["ate"], tau, cfg.near_band)
        near_acc_back.append(na_back)
        dtheta.append(float(_rms(jnp.asarray(flatf(th_back) - flatf(theta_anchor)))))
        weight_back = reparam_alt(th_back, B["knots"], B["scales"], B["mask"])
        dW.append(float(_rms(jnp.asarray(flatf(weight_back) - flatf(weight_anchor)))))


    result = dict(
        tau=tau, step_idx=step_idx, n_bulk_steps=ns, anchor_loss=step_loss,
        anchor_test_acc=overall, anchor_near_tau_acc=near_acc,
        grid=grid, final_loss_to=np.array(final_loss_to),
        final_loss_back=np.array(final_loss_back),
        near_acc_at_read=np.array(near_acc_at_read),
        near_acc_back=np.array(near_acc_back),
        dtheta=np.array(dtheta), dW=np.array(dW),
        cycle=np.array(rec_cycle), tau_c=np.array(rec_tau),
        leg=np.array(rec_leg, dtype="<U8"), loss=np.array(rec_loss),
        theta=np.concatenate(rec_theta, axis=0))

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
        np.savez_compressed(out_path, **result)
        if verbose:
            print(f"saved -> {out_path}")
    return result


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Readout grid sweep anchored on an arbitrary bulk step "
                    "of single-tau training, instead of the refine endpoint.")
    p.add_argument("--step-idx", type=int, required=True,
                   help="bulk step index to anchor the readout on (0 = untrained "
                        "params0; must be <= the number of steps bulk actually ran)")
    p.add_argument("--load", default=None,
                   help="load the bulk trajectory from a run_single_tau .npz "
                        "(must contain bulk_theta/bulk_loss/tau/kink) instead of "
                        "retraining -- pass --img/--channels/--kink-fraction/"
                        "--seed/--seed-kink/--data matching whatever produced it")
    p.add_argument("--img", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--tau", type=float, default=None,
                   help="fixed tau to train to (default: cfg.tau_max, or the "
                        "loaded file's own tau if --load is given)")
    p.add_argument("--grid-half-width", type=float, default=0.3,
                   help="readout tau_read grid spans [tau-w, tau+w] (default 0.3)")
    p.add_argument("--grid-step", type=float, default=0.05,
                   help="readout grid spacing (default 0.05)")
    p.add_argument("--grid-steps", type=int, default=None,
                   help="GD steps per readout leg (default: cfg.budget_steps)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seed-kink", type=int, default=None)
    p.add_argument("--kink-fraction", type=float, default=None,
                   help="fraction of parameters that are kinked (default: cfg's own default)")
    p.add_argument("--out", default="runs/wmam_single_tau_readout_step.npz")
    p.add_argument("--plot", default=None, help="path to save a readout-panel PNG")
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
    if a.img is not None:
        overrides["img"] = a.img
    if a.channels is not None:
        overrides["channels"] = a.channels
    if a.data is not None:
        overrides["data"] = a.data
    cfg = WMAMConfig(**overrides)

    result = readout_at_step(cfg, tau=a.tau, step_idx=a.step_idx, out_path=a.out,
                              grid_half_width=a.grid_half_width, grid_step=a.grid_step,
                              grid_steps=a.grid_steps, load=a.load)
    if a.plot:
        plot_readout_step(result, save_path=a.plot)


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    main()
