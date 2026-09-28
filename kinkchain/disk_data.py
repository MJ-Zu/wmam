"""Save / load the synthetic disk dataset, so a single FIXED dataset can be
reused across runs.

Why: `make_data` is deterministic, but its seed is tied to the run seed, so
sweeping seeds (or batch sizes, or optimizers) also changes the data. Saving one
dataset and loading it into every run holds the data constant, isolating the
effect you are actually studying from data variation.

Generate & save (from the project root, folder containing kinkchain/):

    python -m kinkchain.disk_data --out data/disk_data.npz
    python -m kinkchain.disk_data --out data/disk_hard.npz --property-noise 0.15 --n-train 120

Use it:

    from kinkchain.mam_weights import WMAMConfig, run_wmam
    from kinkchain.disk_data import load_disk_data
    data = load_disk_data("data/disk_data.npz")
    run_wmam(WMAMConfig(), data=data)          # or: run_wmam(cfg, data="data/disk_data.npz")

The saved npz holds X_train, a_train, X_test, a_test plus the generating params.
Set cfg.img / property_noise to match the saved data when you reuse it (the
array shapes drive everything else).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .mam_weights import WMAMConfig, make_data


def save_disk_data(path, img=16, n_train=300, n_test=350, property_noise=0.15, seed=0):
    """Generate the disk dataset and save it to `path` (an .npz)."""
    cfg = WMAMConfig(img=img, n_train=n_train, n_test=n_test,
                     property_noise=property_noise, seed=seed)
    Xtr, atr, Xte, ate = make_data(cfg)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path,
             X_train=np.asarray(Xtr), a_train=np.asarray(atr),
             X_test=np.asarray(Xte), a_test=np.asarray(ate),
             img=img, n_train=n_train, n_test=n_test,
             property_noise=property_noise, seed=seed)
    print(f"saved {n_train} train + {n_test} test images "
          f"({img}x{img}, property_noise={property_noise}, seed={seed}) -> {path}")
    return path


def load_disk_data(path):
    """Load a saved dataset. Returns (X_train, a_train, X_test, a_test) as JAX
    arrays, ready to pass to run_wmam(..., data=...)."""
    import jax.numpy as jnp
    d = np.load(path)
    return (jnp.asarray(d["X_train"]), jnp.asarray(d["a_train"]),
            jnp.asarray(d["X_test"]), jnp.asarray(d["a_test"]))


def describe(path):
    """Print a quick summary + class balance at a few thresholds."""
    d = np.load(path)
    atr, ate = d["a_train"], d["a_test"]
    print(f"{path}: X_train {d['X_train'].shape}, X_test {d['X_test'].shape}, "
          f"property_noise={float(d['property_noise'])}")
    for tau in (0.3, 0.5, 0.7):
        print(f"  tau={tau}: train pos frac={float((atr>=tau).mean()):.2f}  "
              f"test pos frac={float((ate>=tau).mean()):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="save the synthetic disk dataset")
    p.add_argument("--out", default="data/disk_data.npz")
    p.add_argument("--img", type=int, default=16)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-test", type=int, default=5500)
    p.add_argument("--property-noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    save_disk_data(a.out, a.img, a.n_train, a.n_test, a.property_noise, a.seed)
    describe(a.out)
