"""Config load/save wrapper around mam_weights_single_tau_reparam_alt.
run_single_tau (mam_weights_single_tau_reparam_alt.py is not modified).

Why: on the cluster, --seed/--seed-kink are drawn randomly per job (the
SLURM script does seed=$RANDOM, kinkseed=$RANDOM) and folded into the output
.npz filename, but the WMAMConfig itself is never written anywhere -- so
there's no durable record of the FULL config (everything besides seed/
seed_kink) that produced a given run, and no easy way to re-run the exact
same setup except a different seed. This wrapper:

  1. optionally LOADS a base WMAMConfig from a saved JSON file (--load-cfg),
     instead of always starting from WMAMConfig()'s hardcoded defaults;
  2. applies the same CLI overrides mam_weights_single_tau_reparam_alt.py's
     own main() supports (--tau/--seed/--seed-kink/--kink-fraction) ON TOP
     of that loaded base (dataclasses.replace, not a fresh WMAMConfig(...) --
     so fields not being overridden keep the loaded file's values, not the
     dataclass defaults);
  3. SAVES the fully-resolved WMAMConfig actually used (seed and seed_kink
     included) to a JSON file, by default alongside the run's output (same
     path as --out with .npz replaced by _cfg.json) -- so every run's exact
     configuration is reproducible from its own JSON even though seed/
     seed_kink were only decided at submit time.

Run from the project root:
    python -m kinkchain.mam_weights_single_tau_cfg_io --tau 0.55 --seed 15467 --seed-kink 13900 \
        --out runs/wmam_single_tau0.55_kink0.0_15467_13900.npz
    python -m kinkchain.mam_weights_single_tau_cfg_io --load-cfg runs/base_cfg.json --seed 999
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace

import numpy as np

from .mam_weights import _build
from .mam_weights_knot_criterion import count_knot_crossings
from .mam_weights_single_tau_reparam_alt import WMAMConfig, run_single_tau, plot_single_tau


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


def grid_point_crossings(cfg, result, legs=("to", "back")):
    """Which SPECIFIC kinked parameters cross their own knot at each readout
    grid point, for one or both legs.

    `result` is a mam_weights_single_tau_readout_step.readout_at_step return
    value, or its loaded .npz (must have 'grid', 'cycle', 'leg', 'theta' --
    the per-step recording that function's readout grid logs). `cfg` must
    describe the SAME setup that produced `result` (img/channels/
    kink_fraction/seed/seed_kink/data), since knots/scales/the kink mask
    aren't stored in that file and are regenerated via _build(cfg) here --
    same assumption readout_at_step's own --load makes, not re-checked here.

    Returns a dict: `grid` (n_grid,), `kink` (n_params,) bool mask, and per
    leg an `n_crossed_<leg>` (n_grid,) count plus a `crossed_<leg>`
    (n_grid, n_params) bool matrix marking exactly which parameters crossed
    during that grid point's leg (False everywhere for non-special params).
    """
    B = _build(cfg)
    kink_mask = B["kink"]
    knots_flat = B["flat"](B["knots"])

    grid = np.asarray(result["grid"])
    cycle = np.asarray(result["cycle"])
    leg_arr = np.asarray(result["leg"])
    theta = np.asarray(result["theta"])

    out = dict(grid=grid, kink=kink_mask)
    for leg in legs:
        n_crossed = np.zeros(len(grid), dtype=int)
        crossed = np.zeros((len(grid), kink_mask.shape[0]), dtype=bool)
        for idx in range(len(grid)):
            m = (cycle == idx) & (leg_arr == leg)
            if not m.any():
                continue
            nc, cmask = count_knot_crossings(theta[m], knots_flat, kink_mask)
            n_crossed[idx] = nc
            crossed[idx] = cmask
        out[f"n_crossed_{leg}"] = n_crossed
        out[f"crossed_{leg}"] = crossed
    return out


def save_grid_crossings(cfg, result_or_path, out_path=None, verbose=True):
    """grid_point_crossings, saved to disk.

    result_or_path: a readout_at_step result dict, or a path to its saved
    .npz. out_path defaults to result_or_path with a _crossings.npz suffix
    (or runs/grid_crossings.npz if a dict was passed directly)."""
    if isinstance(result_or_path, str):
        result = np.load(result_or_path, allow_pickle=False)
        default_out = os.path.splitext(result_or_path)[0] + "_crossings.npz"
    else:
        result = result_or_path
        default_out = "runs/grid_crossings.npz"
    out_path = out_path or default_out

    out = grid_point_crossings(cfg, result)
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(out_path, **out)
    if verbose:
        for leg in ("to", "back"):
            key = f"n_crossed_{leg}"
            if key in out:
                print(f"[crossings] {leg:5s} leg: n_crossed per grid point = {out[key].tolist()}")
        print(f"[crossings] saved -> {out_path}")
    return out


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="run_single_tau, plus loading a base WMAMConfig from JSON "
                    "and saving the fully-resolved cfg (seed/seed_kink included) "
                    "back out alongside the run's output.")
    p.add_argument("--load-cfg", default=None,
                   help="JSON file (from a prior save_cfg) to use as the base "
                        "cfg instead of WMAMConfig()'s hardcoded defaults")
    p.add_argument("--save-cfg", default=None,
                   help="where to save the resolved cfg (default: --out with "
                        ".npz replaced by _cfg.json)")
    p.add_argument("--no-save-cfg", action="store_true",
                   help="skip saving the resolved cfg")
    p.add_argument("--cfg-only", action="store_true",
                   help="resolve and save the cfg, print it, then exit -- "
                        "skip run_single_tau entirely (no training)")
    p.add_argument("--analyze-crossings", default=None,
                   help="path to an existing readout_at_step .npz (with grid/"
                        "cycle/leg/theta) -- compute which kinked parameters "
                        "cross their knot at each grid point and exit (no "
                        "training); cfg must match whatever produced that file")
    p.add_argument("--crossings-out", default=None,
                   help="where to save --analyze-crossings output (default: "
                        "the input path with a _crossings.npz suffix)")
    p.add_argument("--tau", type=float, default=None,
                   help="fixed tau to train to (default: cfg.tau_max)")
    p.add_argument("--grid-half-width", type=float, default=0.3)
    p.add_argument("--grid-step", type=float, default=0.05)
    p.add_argument("--grid-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seed-kink", type=int, default=None)
    p.add_argument("--kink-fraction", type=float, default=None)
    p.add_argument("--out", default="runs/wmam_single_tau.npz")
    p.add_argument("--plot", default=None, help="path to save a loss-curve PNG")
    return p.parse_args()


def main():
    a = _parse_args()

    base = load_cfg(a.load_cfg) if a.load_cfg else WMAMConfig()
    overrides = {}
    if a.kink_fraction is not None:
        overrides["kink_fraction"] = a.kink_fraction
    if a.seed is not None:
        overrides["seed"] = a.seed
    if a.seed_kink is not None:
        overrides["seed_kink"] = a.seed_kink
    cfg = replace(base, **overrides) if overrides else base

    if not a.no_save_cfg:
        cfg_path = a.save_cfg or _default_cfg_path(a.out)
        save_cfg(cfg, cfg_path)
        print(f"[cfg] saved resolved config (seed={cfg.seed}, seed_kink={cfg.seed_kink}) -> {cfg_path}")

    if a.cfg_only:
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
        return

    if a.analyze_crossings:
        save_grid_crossings(cfg, a.analyze_crossings, out_path=a.crossings_out)
        return

    print(cfg.channels, cfg.seed, cfg.seed_kink)
    phases = run_single_tau(cfg, tau=a.tau, out_path=a.out,
                             grid_half_width=a.grid_half_width, grid_step=a.grid_step,
                             grid_steps=a.grid_steps)
    if a.plot:
        plot_single_tau(phases, a.tau if a.tau is not None else cfg.tau_max, save_path=a.plot)


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    main()
