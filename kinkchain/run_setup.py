"""Run training from a setup file: only an input path and an output path.

    python -m kinkchain.run_setup --setup setups/exp1.setup.pkl --out runs/exp1_bulk.npz
    python -m kinkchain.run_setup --setup configs/exp1_refine.json --out runs/exp1_refine.npz

The setup (see kinkchain/setup_io.py) holds the configuration, the frozen dataset /
init / kinks and the run options; run.mode picks what runs:
  single_tau  bulk -> refine at one tau           (mam_weights_single_tau_train.run_single_tau)
  cyclic      cold + bulk cycles                  (mam_weights_reparam_alt.cyclic_train_logged)
  refine      refine cycles from a previous run   (refine_cyclic_train_logged, needs run.load)
  further     further bulk cycles, same entry     (refine_cyclic_train_logged(further=True))
--load overrides run.load. Outputs stay .npz (loss curves and weight trajectories are
large arrays); next to each output <stem>_cfg.json records the setup path, the
resolved cfg and the run options.
"""

from __future__ import annotations

import argparse
import json
import os

from . import setup_io


def _write_cfg_json(out, setup_path, cfg_dict, run):
    root, _ = os.path.splitext(out)
    d = os.path.dirname(root)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(root + "_cfg.json", "w") as f:
        json.dump({"setup": setup_path, "cfg": cfg_dict, "run": run}, f, indent=2, sort_keys=True)


def main():
    p = argparse.ArgumentParser(description="training from a setup file: input path in, output path out")
    p.add_argument("--setup", required=True, help="setup file: .pkl (frozen) or .json (recipe)")
    p.add_argument("--out", default=None, help="output .npz (default: run.out of the setup)")
    p.add_argument("--load", default=None, help="previous output to continue from (refine / further); overrides run.load")
    a = p.parse_args()

    B, cfg_dict, run = setup_io.load_setup(a.setup)
    mode = run["mode"]
    out = a.out or run.get("out")
    if not out:
        p.error("no output path: pass --out or set run.out in the setup")
    load = a.load or run.get("load")
    run = {**run, "out": out, "load": load}
    _write_cfg_json(out, a.setup, cfg_dict, run)

    if mode == "single_tau":
        from .mam_weights_single_tau_train import WMAMConfig, run_single_tau
        run_single_tau(WMAMConfig(**cfg_dict), tau=run["tau"], out_path=out,
                       grad_clip=run["grad_clip"], setup=B)
        return

    from .mam_weights_reparam_alt import WMAMConfig, cyclic_train_logged, refine_cyclic_train_logged
    cfg = WMAMConfig(**cfg_dict)
    if mode == "cyclic":
        cyclic_train_logged(cfg, setup=B, out_path=out, grad_clip=run["grad_clip"],
                            stride=run["stride"], batch_seed=run["batch_seed"])
    else:
        if not load:
            p.error(f"mode {mode!r} continues a previous run: pass --load or set run.load")
        refine_cyclic_train_logged(cfg, load=load, setup=B, out_path=out, further=(mode == "further"),
                                   rescale_kink=run["rescale"], grad_clip=run["grad_clip"],
                                   stride=run["stride"], batch_seed=run["batch_seed"])


if __name__ == "__main__":
    main()
