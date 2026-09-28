#!/bin/bash
# Re-run the readout probe anchored at intermediate bulk steps of the runs
# produced by scripts/train_single.sh (no retraining of the bulk phase).
# Run from the repository root:
#     bash scripts/readout_steps.sh
set -e
tau=0.55
channel=12
seeds="29693_31408"   # must match the seeds used in train_single.sh

for kink in 0.0 0.2 0.4
do
for step in 10 100 1000 10000 100000
do
python -m kinkchain.mam_weights_single_tau_readout_step_cfg \
	--load-cfg runs/wmam_single_tau${tau}_kink${kink}_img16_channel${channel}_${seeds}_cfg.json \
	--load runs/wmam_single_tau${tau}_kink${kink}_img16_channel${channel}_${seeds}.npz \
	--step-idx ${step} \
	--no-save-cfg \
	--kink-fraction ${kink} \
	--out runs/wmam_single_tau${tau}_kink${kink}_img16_channel${channel}_${seeds}_step${step}.npz
done
done
