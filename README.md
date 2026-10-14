# wmam-single

Train a **tiny CNN at a fixed threshold τ** (or cycling τ up and down) on a "disk" or UTKFace
dataset, with an optional **kinked reparameterization** on a random subset of weights, and probe
whether the trained weights sit on a *marginally absorbing manifold* (MAM).

Everything is written in JAX + Flax (linen) + optax.

## What it does

- **Data.** Each 16×16 image contains a disk whose radius encodes a hidden property
  `a ∈ [0, 1]`; the binary label is `a >= τ`. Fixed datasets (100 train / 150 test) are in
  `data/`: `disk_data_100_150.npz`, `disk_data_100_150_strat.npz` (every 0.01 interval of `a`
  is populated) and `utkface_100_150_64.npz` (64×64 faces, `a` = age).
- **Model.** `TinyCNN`: Conv(3×3) → GELU → Conv(3×3) → GELU → global mean pool → Dense(1),
  with `channels` feature maps.
- **Kinks.** A fraction `kink_fraction` of the parameters enters the network as
  `x' = s·|x − c|` (`kinkchain/reparam.py`, `reparam_alt`): a gradient discontinuity at the
  knot `c`, continuous loss.
- **Training.** `single_tau`: bulk → refine at one τ. `cyclic`: cold start + bulk cycles that
  sweep τ up and down; `further` / `refine`: continue from a previous cyclic output.

## Installation

```bash
git clone https://github.com/MJ-Zu/wmam.git
cd wmam
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run every command from the repository root (the folder that contains `kinkchain/`).

## Setup files: JSON or pickle in, `.npz` out

All settings of a run live in **one setup file**; a training command then only needs the path of
that file and the path of the output.

```bash
# freeze data + initial weights + kinks + config into a pickle (once)
python -m kinkchain.make_setup --config configs/example_cyclic.json --out setups/example_cyclic.setup.pkl
# train: input = setup, output = .npz
python -m kinkchain.run_setup --setup setups/example_cyclic.setup.pkl --out runs/example_bulk.npz
```

| file | what it is |
|---|---|
| `*.setup.pkl` | **frozen**: config, run options, dataset, initial weights, knots, scales, mask. Every machine starts from exactly the same arrays. Arrays are stored as `(dtype, shape, bytes)`, so the file is plain Python objects and independent of the numpy version. Only load pickles you made yourself. |
| `*.json` | **recipe**: `{"base": <optional pickle>, "cfg": {...}, "run": {...}}`. Without `base` the arrays are rebuilt from the data file and seeds on every run. With `base` it is a stage description on a frozen setup: it may change learning rates, step budgets, … but not the fields that define the arrays. |
| `runs/*.npz` | training **output** (loss curves, weight trajectories — big arrays). Next to each, `<stem>_cfg.json` records the setup path, the resolved config and run options. |

`cfg` holds `WMAMConfig` fields (the rest are defaults); `run` holds

| `run` field | meaning |
|---|---|
| `mode` | `single_tau` (bulk → refine at one τ), `cyclic` (cold + bulk cycles), `further` (more bulk cycles from `load`), `refine` (refine cycles from `load`) |
| `tau` | `single_tau` only; default `cfg.tau_max` |
| `grad_clip` | clip every gradient element to `[-c, c]` (`null` = off) |
| `stride` | cyclic modes: record θ and loss every `stride`-th step, inside the training loop |
| `batch_seed` | cyclic modes: seed offset of the mini-batch noise |
| `rescale` | `refine` / `further`: rescale the gradients of kinked parameters |
| `load` | `refine` / `further`: previous output `.npz` to continue from |
| `out` | default output path (`--out` wins) |

`python -m kinkchain.make_setup --show setups/x.setup.pkl` prints what a setup file contains;
`--from-npz old_setup.npz --out new.setup.pkl` converts a setup written by the former `.npz`
`save_setup`.

Examples in `configs/`: `example_cyclic.json` (stage 1), `example_stage2_further.json`,
`example_stage3_refine.json` (layered on the frozen stage-1 setup), `example_single_tau.json`,
`example_utkface_cyclic.json`. `scripts/cyclic_pipeline.sh` runs the three cyclic stages,
`scripts/single_tau.sh` the single-τ run.

Memory: the per-step weight trajectory has `(steps + 1) × n_parameters × 4` bytes (64 channels,
100 000 steps ≈ 14 GiB). Use `"stride": k` (cyclic) or `"log_every": k` (single τ) in the
config to keep every k-th step only.

## Google Colab (GPU)

1. Push this repository to GitHub (`runs/` is git-ignored).
2. In Colab open `notebooks/colab_run.ipynb` from GitHub
   (`https://colab.research.google.com/github/MJ-Zu/wmam/blob/main/notebooks/colab_run.ipynb`),
   set Runtime → Change runtime type → GPU.
3. Run the cells: they clone / pull the repo, install `flax` and `optax`, optionally link `runs/`
   to a Google-Drive folder, then call `make_setup` and `run_setup`.

After changing code or configs locally: commit, push, and `git pull` in the notebook (the clone
cell does it).

## Older entry points

`scripts/train_single.sh` and `scripts/readout_steps.sh` use the flag-based modules
`mam_weights_single_tau_cfg_io` and `mam_weights_single_tau_readout_step_cfg` with
`configs/base_cfg.json`; they are unchanged and still work.

## Repository layout

```
configs/                 base_cfg.json (older scripts) and the example setup recipes
data/                    disk_data_100_150.npz, disk_data_100_150_strat.npz, utkface_100_150_64.npz
notebooks/colab_run.ipynb  Colab driver
scripts/                 shell scripts
setups/                  frozen *.setup.pkl (make_setup output)
kinkchain/
  setup_io.py            read / write setup files (JSON, pickle)
  make_setup.py          JSON recipe -> frozen pickle (--show, --from-npz)
  run_setup.py           train from a setup file: --setup IN --out OUT
  mam_weights.py                   TinyCNN, config, data, single-τ training blocks
  mam_weights_single_tau_train.py  bulk -> refine at one τ (run_single_tau)
  mam_weights_reparam_alt.py       cyclic training, in-loop interval logging
  mam_weights_single_tau_reparam_alt.py, ..._readout_step*.py, mam_weights_knot_criterion.py
                                   readout / knot-crossing analysis (older entry points)
  reparam.py, mam.py               kinked reparameterization, shared helpers
  disk_data.py, make_disk_stratified.py, make_utkface_subset.py   data generation
  utkface_feasibility.py           learning-rate / τ feasibility check on UTKFace
```

Data generation:

```bash
python -m kinkchain.disk_data --out data/my_data.npz --n-train 400 --n-test 450
python -m kinkchain.make_disk_stratified --out data/my_strat.npz   # every 0.01 bin of a filled
```

## Author

Mengjie Zu (IST Austria)
