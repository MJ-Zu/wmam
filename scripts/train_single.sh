#!/bin/bash
# Train the tiny CNN at a single fixed tau (bulk -> refine -> readout),
# once per kink fraction. Run from the repository root:
#     bash scripts/train_single.sh
set -e
tau=0.55
channel=12
seed=29693        # or: seed=$RANDOM
seedkink=31408    # or: seedkink=$RANDOM

mkdir -p runs
for kink in 0.0 0.2 0.4
do
python -m kinkchain.mam_weights_single_tau_cfg_io \
	--tau ${tau} \
	--kink-fraction ${kink} \
	--seed ${seed} \
	--seed-kink ${seedkink} \
	--load-cfg configs/base_cfg.json \
	--save-cfg runs/wmam_single_tau${tau}_kink${kink}_img16_channel${channel}_${seed}_${seedkink}_cfg.json \
	--out runs/wmam_single_tau${tau}_kink${kink}_img16_channel${channel}_${seed}_${seedkink}.npz
done
