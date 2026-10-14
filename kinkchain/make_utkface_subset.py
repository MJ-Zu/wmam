"""Build a small UTKFace subset in the disk-data npz format.

Source: data/utkface_128.npz (images uint8 N x 128 x 128 x 3, ages int).
Ages 15..64 (50 integer values); for each age, 5 distinct images are drawn
without replacement (fixed seed): the first 2 go to train, the other 3 to test,
so train and test never overlap -> 100 train / 150 test.

Preprocessing mirrors make_data's convention:
  * RGB -> gray (0.299, 0.587, 0.114), 128 -> 64 by 2x2 average pooling
  * standardise with the mean/std over train+test together:
    X = (X - mean) / (std + 1e-6)
The property `a` is the raw integer age stored as float32 (label = a >= tau).

The saved keys are those of disk_data.save_disk_data (X_train, a_train, X_test,
a_test, img, n_train, n_test, property_noise, seed), so
disk_data.load_disk_data reads it unchanged. Two extra keys (src_idx_train,
src_idx_test: row indices into the source file) are kept for traceability.

Run from the project root:
    python -m kinkchain.make_utkface_subset
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def to_gray_pooled(images_u8):
    """(N,128,128,3) uint8 -> (N,64,64) float32: RGB->gray, then 2x2 mean pool."""
    x = images_u8.astype(np.float32)
    gray = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
    n, h, w = gray.shape
    return gray.reshape(n, h // 2, 2, w // 2, 2).mean(axis=(2, 4)).astype(np.float32)


def build_subset(src, out, seed=0, age_min=15, age_max=64, n_train_per_age=2, n_test_per_age=3):
    z = np.load(src, allow_pickle=False)
    images, ages = z["images"], z["ages"]
    rng = np.random.default_rng(seed)
    per_age = n_train_per_age + n_test_per_age

    tr_idx, te_idx = [], []
    for age in range(age_min, age_max + 1):
        pool = np.where(ages == age)[0]
        if len(pool) < per_age:
            raise ValueError(f"age {age}: only {len(pool)} images, need {per_age}")
        pick = rng.choice(pool, per_age, replace=False)
        tr_idx.extend(pick[:n_train_per_age].tolist())
        te_idx.extend(pick[n_train_per_age:].tolist())
    tr_idx = np.array(tr_idx)[rng.permutation(len(tr_idx))]
    te_idx = np.array(te_idx)[rng.permutation(len(te_idx))]
    assert len(set(tr_idx.tolist()) & set(te_idx.tolist())) == 0, "train/test overlap"

    Xtr = to_gray_pooled(images[tr_idx])
    Xte = to_gray_pooled(images[te_idx])
    allx = np.concatenate([Xtr, Xte])
    mean, std = allx.mean(dtype=np.float64), allx.std(dtype=np.float64)
    Xtr = ((Xtr - mean) / (std + 1e-6)).astype(np.float32)[..., None]
    Xte = ((Xte - mean) / (std + 1e-6)).astype(np.float32)[..., None]
    atr = ages[tr_idx].astype(np.float32)
    ate = ages[te_idx].astype(np.float32)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, X_train=Xtr, a_train=atr, X_test=Xte, a_test=ate,
             img=64, n_train=len(tr_idx), n_test=len(te_idx),
             property_noise=0.0, seed=seed,
             src_idx_train=tr_idx.astype(np.int64), src_idx_test=te_idx.astype(np.int64))
    print(f"saved {len(tr_idx)} train + {len(te_idx)} test ({Xtr.shape[1]}x{Xtr.shape[2]}, "
          f"seed={seed}) -> {out}")
    print(f"standardisation: mean={mean:.4f} std={std:.4f}  "
          f"(result: mean={np.concatenate([Xtr, Xte]).mean():.2e}, std={np.concatenate([Xtr, Xte]).std():.4f})")
    return out


def describe(path, taus=(38, 39, 40, 41, 42, 43)):
    d = np.load(path)
    atr, ate = d["a_train"], d["a_test"]
    print(f"\n{path}: X_train {d['X_train'].shape} {d['X_train'].dtype}, "
          f"X_test {d['X_test'].shape}, img={int(d['img'])}, "
          f"property_noise={float(d['property_noise'])}")
    ages = np.arange(int(atr.min()), int(atr.max()) + 1)
    ctr = np.array([(atr == a).sum() for a in ages])
    cte = np.array([(ate == a).sum() for a in ages])
    print(f"per-age counts over {len(ages)} ages ({ages[0]}..{ages[-1]}): "
          f"train min/max = {ctr.min()}/{ctr.max()}, test min/max = {cte.min()}/{cte.max()}")
    print("age : train test")
    for i in range(0, len(ages), 10):
        print("  " + "  ".join(f"{a}:{ctr[j]}/{cte[j]}" for j, a in zip(range(i, i + 10), ages[i:i + 10])))
    print("\ntau   train pos frac (n_pos)   test pos frac (n_pos)")
    for t in taus:
        print(f"{t:3d}   {float((atr >= t).mean()):.2f} ({int((atr >= t).sum()):3d})"
              f"            {float((ate >= t).mean()):.2f} ({int((ate >= t).sum()):3d})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="UTKFace 100/150 subset in disk-data format")
    p.add_argument("--src", default="data/utkface_128.npz")
    p.add_argument("--out", default="data/utkface_100_150_64.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--age-min", type=int, default=15)
    p.add_argument("--age-max", type=int, default=64)
    a = p.parse_args()
    build_subset(a.src, a.out, a.seed, a.age_min, a.age_max)
    describe(a.out)
