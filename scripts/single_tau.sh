#!/bin/bash
# Bulk -> refine at one fixed tau from a setup file. Run from the repository root:
#     bash scripts/single_tau.sh
set -e
mkdir -p runs
python -m kinkchain.run_setup --setup configs/example_single_tau.json --out runs/example_single_tau.npz
