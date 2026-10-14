#!/bin/bash
# Cyclic training in three stages, driven by setup files: every command only needs an input
# (setup) path and an output path. Run from the repository root:
#     bash scripts/cyclic_pipeline.sh
set -e
mkdir -p runs setups

# freeze the data / init / kinks once (pickle); the same file gives the same start on any machine
python -m kinkchain.make_setup --config configs/example_cyclic.json --out setups/example_cyclic.setup.pkl

# stage 1: cold + bulk cycles
python -m kinkchain.run_setup --setup setups/example_cyclic.setup.pkl      --out runs/example_bulk.npz
# stage 2: further bulk cycles (JSON layered on the frozen setup; run.load = stage 1 output)
python -m kinkchain.run_setup --setup configs/example_stage2_further.json  --out runs/example_further.npz
# stage 3: refine with plain GD (run.load = stage 2 output)
python -m kinkchain.run_setup --setup configs/example_stage3_refine.json   --out runs/example_refine.npz
