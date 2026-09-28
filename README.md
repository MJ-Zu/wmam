# wmam-single

Train a **tiny CNN at a single fixed threshold τ** on a synthetic "disk" dataset, with an
optional **kinked reparameterization** on a random subset of weights, and probe whether the
trained weights sit on a *marginally absorbing manifold* (MAM) via a round-trip readout.

Everything is written in JAX + Flax (linen) + optax.

## What it does

- **Data.** Each 16×16 image contains a disk whose radius encodes a hidden property
  `a ∈ [0, 1]`. The binary label is `a >= τ`. A fixed dataset (100 train / 150 test) is
  included in `data/disk_data_100_150.npz`.
- **Model.** `TinyCNN`: Conv(3×3) → GELU → Conv(3×3) → GELU → global mean pool →
  Dense(1), with `channels` feature maps (12 by default).
- **Kinks.** A fraction `kink_fraction` of the parameters is fed to the network as
  `x' = s·|x − c|` (see `kinkchain/reparam.py`, `reparam_alt`). This puts a controlled
  gradient discontinuity at the knot `c` while leaving the loss continuous.
- **Three training phases at one τ** (`kinkchain/mam_weights_single_tau_reparam_alt.py`):
  1. **bulk**: optimizer from `bulk_optimizer`, cosine-decayed lr, cold start;
  2. **refine**: plain gradient descent, warm-started from the end of bulk;
  3. **readout**: for every `tau_read` on a grid around τ, retrain to `tau_read` and
     back to τ, then measure near-boundary accuracy and weight displacement.

## Installation

```bash
git clone https://github.com/<your-username>/wmam-single.git
cd wmam-single
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run every command from the repository root (the folder that contains `kinkchain/`).

**Train** (one run per kink fraction 0.0 / 0.2 / 0.4):

```bash
bash scripts/train_single.sh
```

Each run writes `runs/<name>.npz` (loss curves, weight trajectory, readout results) and
`runs/<name>_cfg.json` (the full configuration used, so the run can be reproduced exactly).

**Readout at intermediate bulk steps** (reuses the saved trajectory, no retraining):

```bash
bash scripts/readout_steps.sh
```

**Single run by hand:**

```bash
python -m kinkchain.mam_weights_single_tau_cfg_io \
    --load-cfg configs/base_cfg.json \
    --tau 0.55 --kink-fraction 0.2 --seed 1 --seed-kink 2 \
    --out runs/my_run.npz --plot runs/my_run.png
```

Add `--help` to either module to list all options.

## Configuration

`configs/base_cfg.json` holds the base hyper-parameters (image size, channels, learning
rates, step budgets, dataset path, …). Command-line flags such as `--seed`, `--seed-kink`
and `--kink-fraction` override the matching fields. The fully resolved configuration is
always saved next to the output. Field meanings are documented on `WMAMConfig` in
`kinkchain/mam_weights.py`.

With the default `max_steps = 100000`, one run takes a while on a CPU. For a quick check,
copy the config and lower `max_steps`.

## Repository layout

```
configs/base_cfg.json      base hyper-parameters
data/disk_data_100_150.npz fixed synthetic dataset
scripts/                   shell scripts for training and readout
kinkchain/
  mam_weights.py                           TinyCNN, WMAMConfig, data, training blocks
  mam_weights_single_tau_reparam_alt.py    bulk -> refine -> readout at one τ
  mam_weights_single_tau_cfg_io.py         entry point: train with JSON config in/out
  mam_weights_single_tau_readout_step.py   readout anchored at a given bulk step
  mam_weights_single_tau_readout_step_cfg.py  entry point for the above, JSON config
  mam_weights_reparam_alt.py, reparam.py   kinked reparameterization
  mam_weights_knot_criterion.py            counting knot crossings
  mam.py, disk_data.py                     shared helpers, dataset save/load
```

To generate a different dataset:

```bash
python -m kinkchain.disk_data --out data/my_data.npz --n-train 400 --n-test 450
```

## Author

Mengjie Zu (IST Austria)
