"""Disk dataset whose property `a` covers [0, 1) evenly: every interval
[k*bin, (k+1)*bin) (default bin = 0.01) holds at least one training image and at
least one test image.

Why: with a ~ U(0,1) i.i.d. (make_data / disk_data.py) a 100-image training set
leaves about a third of the 0.01-bins empty. The label is (a >= tau), so moving
tau by 0.01 across an empty bin flips NO training label: the loss, and so the
weights, do not change between those two tau values, and a readout with
increment 0.01 shows identical results at neighbouring tau_read. Here the base
layer is one image per bin (so, with n_train = 100, exactly one label flips per
0.01 step in tau); any extra images (n > 1/bin) are i.i.d. uniform on [0, 1).

The images are rendered exactly like make_data (disk radius (0.12 + 0.45 a) * H at
a random centre, N(0, 0.1) pixel noise, standardised over train+test together);
only the way `a` is drawn differs. The saved keys are those of
disk_data.save_disk_data (plus strat_bin), so disk_data.load_disk_data reads the
file unchanged.

Run from the project root:
    python -m kinkchain.make_disk_stratified
    python -m kinkchain.make_disk_stratified --out data/my_disk.npz --seed 3 --n-train 200 --n-test 300
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def stratified_a(n, rng, bin_width=0.01, margin=0.05):
    """n values in [0, 1): one per bin of width `bin_width` (placed at least
    `margin` * bin_width away from the bin edges, so float32 rounding can never
    push it into a neighbouring bin), plus n - n_bins i.i.d. uniform extras."""
    n_bins = int(round(1.0 / bin_width))
    if n < n_bins:
        raise ValueError(f"need at least {n_bins} images to fill every {bin_width} bin, got {n}")
    base = (np.arange(n_bins) + rng.uniform(margin, 1.0 - margin, n_bins)) * bin_width
    extra = rng.uniform(0.0, 1.0, n - n_bins)
    return rng.permutation(np.concatenate([base, extra]))


def render_images(a, img, property_noise, rng):
    """make_data's image loop: a (N,) -> standardised images (N, img, img, 1)."""
    N = len(a)
    a_img = np.clip(a + rng.normal(0.0, property_noise, N), 0.0, 1.0)
    yy, xx = np.mgrid[0:img, 0:img]
    X = np.zeros((N, img, img, 1), np.float32)
    for i in range(N):
        cy, cx = rng.uniform(0.35, 0.65, 2) * img
        R = (0.12 + 0.45 * a_img[i]) * img
        disk = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= R).astype(np.float32)
        X[i, :, :, 0] = np.clip(disk + rng.normal(0, 0.1, (img, img)), 0.0, 1.5)
    return (X - X.mean()) / (X.std() + 1e-6)


def bin_counts(a, bin_width=0.01):
    n_bins = int(round(1.0 / bin_width))
    return np.bincount(np.minimum((np.asarray(a, np.float64) / bin_width).astype(int), n_bins - 1), minlength=n_bins)


def build(out, n_train=100, n_test=150, img=16, property_noise=0.0, seed=0, bin_width=0.01):
    rng = np.random.default_rng(seed + 7)
    a_tr = stratified_a(n_train, rng, bin_width)
    a_te = stratified_a(n_test, rng, bin_width)
    a = np.concatenate([a_tr, a_te])
    X = render_images(a, img, property_noise, rng)
    assert bin_counts(a_tr, bin_width).min() >= 1 and bin_counts(a_te, bin_width).min() >= 1

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, X_train=X[:n_train], a_train=a_tr.astype(np.float32),
             X_test=X[n_train:], a_test=a_te.astype(np.float32),
             img=img, n_train=n_train, n_test=n_test, property_noise=property_noise,
             seed=seed, strat_bin=bin_width)
    print(f"saved {n_train} train + {n_test} test ({img}x{img}, seed={seed}, every {bin_width} bin filled) -> {out}")
    return out


def describe(path, bin_width=0.01, taus=np.round(np.arange(0.45, 0.6001, 0.01), 6)):
    d = np.load(path)
    atr, ate = d["a_train"].astype(np.float64), d["a_test"].astype(np.float64)
    ctr, cte = bin_counts(atr, bin_width), bin_counts(ate, bin_width)
    print(f"\n{path}: X_train {d['X_train'].shape}, X_test {d['X_test'].shape}")
    print(f"training images per {bin_width} bin over [0, 1): min {ctr.min()}, max {ctr.max()}, empty bins {(ctr == 0).sum()}"
          f" | test: min {cte.min()}, max {cte.max()}, empty bins {(cte == 0).sum()}")
    print("training images per 0.01 bin, bins 0.42 .. 0.57 (= labels that flip when tau moves up by 0.01):")
    print("  " + "  ".join(f"{k / 100:.2f}:{ctr[k]}" for k in range(42, 58)))
    labels = np.array([(atr >= t) for t in taus])
    print(f"distinct training label vectors over tau = {taus[0]:.2f}..{taus[-1]:.2f} step 0.01: "
          f"{len({row.tobytes() for row in labels})} of {len(taus)}")
    print("train positive fraction at tau 0.50 / 0.53 / 0.55: " + " / ".join(f"{(atr >= t).mean():.2f}" for t in (0.50, 0.53, 0.55)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="disk dataset with every 0.01 bin of `a` filled")
    p.add_argument("--out", default="data/disk_data_100_150_strat.npz")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=150)
    p.add_argument("--img", type=int, default=16)
    p.add_argument("--property-noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bin", type=float, default=0.01, help="interval width that must contain data")
    a = p.parse_args()
    build(a.out, a.n_train, a.n_test, a.img, a.property_noise, a.seed, a.bin)
    describe(a.out, a.bin)
