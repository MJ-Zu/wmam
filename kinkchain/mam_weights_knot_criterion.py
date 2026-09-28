"""Knot-crossing count, and a cyclic-training termination criterion built on
it. mam_weights.py is not modified -- this is a new file.

mam_weights.cyclic_train_logged declares "MAM approached" using per-target
loss convergence (`converge_tol`) plus a dD-plateau heuristic on the overall
weight-vector movement ([mam_weights.py:432-438](mam_weights.py)). This
module offers a more direct, mechanistic alternative: a special (kinked)
parameter only "engages" with its kink if it actually crosses its knot `c`
(the sign of `theta - c` flips) at some point during training -- if it never
crosses, it just sits on one side of the corner behaving like an ordinary
(if steeper/shallower) smooth parameter for that whole cycle. Once NOT ONE
special parameter crosses its knot anywhere during a full bulk cycle, every
kinked parameter has settled and stopped chattering; combined with every
target's loss already being below a threshold, that's a natural "the cyclic
phase is done" signal.

Run from the project root:
    python -m kinkchain.mam_weights_knot_criterion --loss-threshold 0.02
    python -m kinkchain.mam_weights_knot_criterion --setup runs/wmam_setup_nokink.npz --loss-threshold 0.02
"""

from __future__ import annotations

import os

import jax
import numpy as np

from .mam_weights import (
    WMAMConfig, _build, make_logging_trainer, _truncate, _triangle,
    _run_cycle_logged, save_log, save_setup, load_setup,
)
from .mam import _rms


def count_knot_crossings(theta_traj, knots_flat, kink_mask):
    """How many SPECIAL parameters cross their own knot at least once across
    a logged (n_steps, n_params) trajectory.

    A crossing is a sign change of (theta - c) between two consecutive
    logged steps: the parameter was on one side of its kink and ended up on
    the other at some point. Non-special parameters (kink_mask False) are
    ignored. A parameter sitting exactly on its knot (diff == 0) contributes
    sign 0; a transition into/out of exactly 0 counts as a change too, since
    np.sign(0) == 0 differs from +-1.

    Args:
        theta_traj: (n_steps, n_params) array -- one row per logged step.
        knots_flat: (n_params,) flattened knot values c (same layout as
            theta_traj's columns, e.g. from B["flat"](B["knots"])).
        kink_mask: (n_params,) boolean, True where the parameter is special
            (e.g. B["kink"]).

    Returns:
        (n_crossed, crossed_mask): the count of distinct special parameters
        that crossed at least once, and a boolean (n_params,) array marking
        exactly which ones (False everywhere for non-special parameters).
    """
    theta_traj = np.asarray(theta_traj)
    knots_flat = np.asarray(knots_flat)
    kink_mask = np.asarray(kink_mask)

    special_idx = np.where(kink_mask)[0]
    diff = theta_traj[:, special_idx] - knots_flat[special_idx]   # (n_steps, n_special)
    sign = np.sign(diff)
    changed = np.any(sign[1:] != sign[:-1], axis=0) if len(sign) > 1 else np.zeros(
        special_idx.shape[0], dtype=bool)

    crossed_mask = np.zeros(kink_mask.shape, dtype=bool)
    crossed_mask[special_idx] = changed
    return int(changed.sum()), crossed_mask


def cyclic_train_knot_criterion(cfg=None, loss_threshold=None, verbose=True,
                                 out_path="runs/wmam_cyclic_knot.npz",
                                 setup=None, stride=1):
    """cyclic_train_logged, with the MAM-approach gate replaced.

    A cycle counts as DONE when:
      (1) every target's final loss in the cycle is below `loss_threshold`
          (defaults to cfg.converge_tol if not given), AND
      (2) not one special parameter crossed its knot anywhere during the
          cycle (count_knot_crossings(...) == 0).
    As soon as one cycle satisfies both, training stops there -- no
    patience/hysteresis like the original's dD_patience; add your own
    "require N consecutive cycles" wrapper around this if you want that.

    Returns the same shape of log dict as cyclic_train_logged, plus
    `knot_crossings` (per-cycle count) and `loss_threshold`.
    """
    cfg = cfg or WMAMConfig()
    loss_threshold = cfg.converge_tol if loss_threshold is None else loss_threshold
    B = setup if setup is not None else _build(cfg)
    common = dict(model=B["model"], knots=B["knots"], scales=B["scales"],
                  Xtr=B["Xtr"], atr=B["atr"], Xte=B["Xte"], ate=B["ate"])
    bulk_log, _, full_loss_valid, _ = make_logging_trainer(
        **common, max_steps=cfg.max_steps, tol=cfg.tol,
        opt_name=cfg.bulk_optimizer, lr=cfg.lr, afactor=cfg.bulk_afactor,
        batch=cfg.batch_size, augment=cfg.augment)
    cold_log, _, _, _ = make_logging_trainer(
        **common, max_steps=cfg.readout_max_steps, tol=cfg.readout_tol,
        opt_name="gd", lr=cfg.readout_lr, afactor=1.0)

    knots_flat = B["flat"](B["knots"])
    kink_mask = B["kink"]
    flatf = B["flat"]

    cyc_c, tau_c, leg_c, loss_c, th_rows = [], [], [], [], []

    def rec(bt, bl, ns, cyc, tau, leg):
        bt, bl = _truncate(bt, bl, ns, stride)
        k = len(bl)
        cyc_c.extend([cyc] * k); tau_c.extend([tau] * k)
        leg_c.extend([leg] * k); loss_c.extend(bl.tolist()); th_rows.append(bt)

    dummy = jax.random.PRNGKey(0)
    down, up = _triangle(cfg)
    key = jax.random.PRNGKey(cfg.seed + 1234)

    theta, ns, lc, bt, bl = cold_log(B["params0"], cfg.tau_max, dummy)
    rec(bt, bl, ns, 0, cfg.tau_max, "init")
    theta_cold = theta
    if verbose:
        print(f"loss_cold={float(lc):.2e}  loss_threshold={loss_threshold:.2e}")

    dD, prev, theta_min, mam_cycle = [], theta, theta, -1
    knot_crossings = []
    for cyc in range(1, cfg.n_cycles + 1):
        rows_before = len(th_rows)
        theta, theta_min, ml, key = _run_cycle_logged(
            bulk_log, theta, down, up, key, rec, cyc, stride)
        dD.append(_rms(np.asarray(flatf(theta) - flatf(prev)))); prev = theta

        cycle_theta = np.concatenate(th_rows[rows_before:], axis=0)
        n_crossed, crossed_mask = count_knot_crossings(cycle_theta, knots_flat, kink_mask)
        knot_crossings.append(n_crossed)

        converged = ml < loss_threshold
        done = converged and (n_crossed == 0)
        vl = full_loss_valid(theta, up[-1])
        if verbose and (cyc == 1 or cyc % 3 == 0 or done):
            print(f"  bulk {cyc:3d}  dD={dD[-1]:.2e}  max_loss={ml:.2e}  "
                  f"converged={converged}  knot_crossings={n_crossed}  "
                  f"valid_loss={vl:.2e}")
        if done:
            mam_cycle = cyc
            if verbose:
                print(f"  -> done at bulk cycle {cyc}: max_loss={ml:.2e} < "
                      f"{loss_threshold:.2e}, and 0/{int(kink_mask.sum())} "
                      f"special parameters crossed their knot")
            break
    n_bulk = len(dD)
    theta_max = theta

    log = dict(
        kind="cyclic_knot", seed=cfg.seed,
        cycle=np.array(cyc_c), tau=np.array(tau_c),
        leg=np.array(leg_c, dtype="<U8"), loss=np.array(loss_c),
        theta=np.concatenate(th_rows, axis=0), kink=B["kink"],
        theta_cold=flatf(theta_cold), theta_max=flatf(theta_max),
        theta_min=flatf(theta_min), dD=np.array(dD),
        knot_crossings=np.array(knot_crossings),
        n_bulk=n_bulk, mam_cycle=mam_cycle, loss_threshold=loss_threshold)
    if out_path:
        save_log(log, out_path, meta=dict(cfg=cfg.as_dict()))
        if verbose:
            print(f"[cyclic-knot] {log['theta'].shape[0]} steps logged -> {out_path}")
    log["_theta_max_tree"] = theta_max
    log["_theta_min_tree"] = theta_min
    return log


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Cyclic training terminated by a knot-crossing "
                    "criterion: done once every target's loss is below "
                    "--loss-threshold AND no special parameter crosses its "
                    "knot during the cycle.")
    p.add_argument("--loss-threshold", type=float, default=None,
                   help="per-cycle max-loss threshold (default: cfg.converge_tol)")
    p.add_argument("--out", default="runs/wmam_cyclic_knot.npz")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--setup", default=None,
                   help="path to a frozen setup .npz (from mam_weights.save_setup): "
                        "loaded if it exists, else built from cfg and saved there")
    return p.parse_args()


def main():
    a = _parse_args()
    cfg = WMAMConfig() if a.seed is None else WMAMConfig(seed=a.seed)

    B = None
    if a.setup:
        if os.path.exists(a.setup):
            B, cfg = load_setup(a.setup)
            print(f"[setup] loaded frozen setup from {a.setup}")
        else:
            B = _build(cfg)
            save_setup(B, cfg, a.setup)
            print(f"[setup] built and froze setup -> {a.setup}")

    cyclic_train_knot_criterion(cfg, loss_threshold=a.loss_threshold,
                                 setup=B, stride=a.stride, out_path=a.out)


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    main()
