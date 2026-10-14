"""Freeze a setup: JSON recipe -> pickle (dataset, init, kinks, cfg, run options).

    python -m kinkchain.make_setup --config configs/example_cyclic.json --out setups/exp1.setup.pkl
    python -m kinkchain.make_setup --from-npz runs/wmam_setup_kink0.4_27140.npz --out setups/old.setup.pkl
    python -m kinkchain.make_setup --show setups/exp1.setup.pkl

Besides the pickle, <stem>.cfg.json is written: a readable copy of the cfg and run
options inside it. Training then needs only the pickle and an output path
(python -m kinkchain.run_setup --setup ... --out ...); see kinkchain/setup_io.py for
the file formats. --from-npz converts a setup written by the old .npz save_setup
(cyclic scripts); --config then supplies the run options for it, if wanted.
"""

from __future__ import annotations

import argparse
import json

from . import setup_io


def main():
    p = argparse.ArgumentParser(description="freeze a training setup into a pickle")
    p.add_argument("--config", help="JSON recipe (see setup_io.py); with --from-npz only its \"run\" block is used")
    p.add_argument("--from-npz", help="old .npz setup to convert")
    p.add_argument("--out", help="pickle to write, e.g. setups/exp1.setup.pkl")
    p.add_argument("--no-json", action="store_true", help="do not write the <stem>.cfg.json copy")
    p.add_argument("--show", help="print a summary of a setup file and exit")
    a = p.parse_args()

    if a.show:
        print(setup_io.describe(a.show))
        return
    if not a.out:
        p.error("--out is required")
    if a.from_npz:
        run = None
        if a.config:
            with open(a.config) as f:
                run = json.load(f).get("run")
        setup_io.npz_to_setup(a.from_npz, a.out, run)
        print(f"converted {a.from_npz} -> {a.out}")
        return
    if not a.config:
        p.error("give --config or --from-npz")
    with open(a.config) as f:
        if "base" in json.load(f):
            p.error(f"{a.config} has a \"base\": it is a stage description on a frozen setup, there is nothing to freeze")
    B, cfg_dict, run = setup_io.load_setup(a.config)
    setup_io.save_setup(B, cfg_dict, run, a.out, json_copy=not a.no_json)
    print(f"froze {a.config} -> {a.out}  (mode={run['mode']}, "
          f"{len(B['kink'])} parameters, {int(B['kink'].sum())} kinked, "
          f"{B['Xtr'].shape[0]} train / {B['Xte'].shape[0]} test images)")


if __name__ == "__main__":
    main()
