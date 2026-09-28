"""readout_at_step, plus loading its WMAMConfig from a saved JSON file.
Neither mam_weights_single_tau_readout_step.py nor mam_weights_single_tau_cfg_io.py
is modified.

mam_weights_single_tau_readout_step.py's own CLI takes cfg overrides one flag
at a time (--img/--channels/--kink-fraction/--seed/--seed-kink/--data), and
you have to get every one of them right to match whatever --load bulk
trajectory file you're pointing at -- exactly the trap from a few turns back,
where --channels 6 (matching a stale filename) would have silently applied
the wrong reparam had the kink-mask-shape check not caught it.

Since mam_weights_single_tau_cfg_io.py now writes a full _cfg.json next to
every run's output, this module lets you point --load-cfg at THAT file
instead of retyping the individual flags -- reconstructing the exact
img/channels/kink_fraction/seed/seed_kink/data (and everything else) that
produced the bulk trajectory in one shot. Any individual override flag can
still be passed on top, same load-then-override pattern as
mam_weights_single_tau_cfg_io.py (dataclasses.replace, not a fresh
WMAMConfig(...), so fields you don't override keep the loaded file's values).
It also saves the resolved cfg back out (same convention as cfg_io), so a
readout-at-step run gets its own paper trail too.

Run from the project root:
    python -m kinkchain.mam_weights_single_tau_readout_step_cfg \
        --load-cfg runs/wmam_single_tau0.55_kink0.0_15467_13900_cfg.json \
        --load runs/wmam_single_tau_0.55_kink0.0_img16_channel12_alt.npz \
        --step-idx 100
"""

from __future__ import annotations

import os
from dataclasses import replace

from .mam_weights_single_tau_reparam_alt import WMAMConfig
from .mam_weights_single_tau_cfg_io import load_cfg, save_cfg, _default_cfg_path
from .mam_weights_single_tau_readout_step import readout_at_step


def plot_readout_step(result, save_path=None):
    """Near-tau accuracy after round-trip (left axis) and weight
    displacement dW (right axis) vs tau_read. Self-contained here because
    mam_weights_single_tau_readout_step.py no longer defines this function
    itself (its own main() still calls it, so that file's --plot is
    currently broken too -- not touching that file, just not depending on
    the missing function)."""
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    grid = result["grid"]
    ax.plot(grid, result["near_acc_back"], color="C2", marker="o", ms=3, lw=1.2,
            label="near-tau accuracy after round-trip")
    ax.axvline(result["tau"], color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("tau_read")
    ax.set_ylabel("near-tau accuracy", color="C2")
    ax.set_ylim(-0.02, 1.02)
    at = ax.twinx()
    at.semilogy(grid, result["dW"], color="C3", marker="s", ms=3, lw=1.0,
                label="weight displacement dW")
    at.set_ylabel("dW (log)", color="C3")
    ax.set_title(f"readout at bulk step {result['step_idx']}/{result['n_bulk_steps']}  "
                f"(anchor_loss={result['anchor_loss']:.2e})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"saved plot -> {save_path}")
    return fig


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="readout_at_step, with its WMAMConfig loadable from a "
                    "saved JSON file (mam_weights_single_tau_cfg_io.save_cfg) "
                    "instead of individually-typed override flags.")
    p.add_argument("--load-cfg", default=None,
                   help="JSON file (from mam_weights_single_tau_cfg_io.save_cfg) "
                        "to use as the base cfg, instead of WMAMConfig()'s defaults")
    p.add_argument("--save-cfg", default=None,
                   help="where to save the resolved cfg (default: --out with "
                        ".npz replaced by _cfg.json)")
    p.add_argument("--no-save-cfg", action="store_true",
                   help="skip saving the resolved cfg")
    p.add_argument("--step-idx", type=int, required=True,
                   help="bulk step index to anchor the readout on")
    p.add_argument("--load", default=None,
                   help="saved run_single_tau .npz (bulk_theta/bulk_loss/tau/"
                        "kink) to read the bulk trajectory from, instead of "
                        "retraining")
    p.add_argument("--img", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--tau", type=float, default=None,
                   help="fixed tau (default: cfg.tau_max, or the loaded "
                        "file's own tau if --load is given)")
    p.add_argument("--grid-half-width", type=float, default=0.3)
    p.add_argument("--grid-step", type=float, default=0.05)
    p.add_argument("--grid-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seed-kink", type=int, default=None)
    p.add_argument("--kink-fraction", type=float, default=None)
    p.add_argument("--out", default="runs/wmam_single_tau_readout_step.npz")
    p.add_argument("--plot", default=None, help="path to save a readout-panel PNG")
    return p.parse_args()


def main():
    a = _parse_args()

    base = load_cfg(a.load_cfg) if a.load_cfg else WMAMConfig()
    overrides = {}
    if a.img is not None:
        overrides["img"] = a.img
    if a.channels is not None:
        overrides["channels"] = a.channels
    if a.data is not None:
        overrides["data"] = a.data
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

    result = readout_at_step(cfg, tau=a.tau, step_idx=a.step_idx, out_path=a.out,
                              grid_half_width=a.grid_half_width, grid_step=a.grid_step,
                              grid_steps=a.grid_steps, load=a.load)
    if a.plot:
        plot_readout_step(result, save_path=a.plot)


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    main()
