"""Feasibility check: can the TinyCNN fit the UTKFace 100-image subset at all?

No kinks (kink_fraction=0 -> reparam is the identity), random initialisation,
full batch. For each tau (default 38..43) the network is trained from scratch
on the label (age >= tau) with Adam, and we record
  * the full-train-loss curve (is it close to 0, e.g. < 1e-6? after how many steps?)
  * train accuracy at the end
  * test accuracy overall and binned by distance |age - tau| to the threshold
    (default bins 0-2, 3-5, 6-10, >10 years): far from the threshold the label
    is easy to read off a face, near it only noise is left, so accuracy far from
    tau is the check for a real signal.

Built on mam_weights_reparam_alt (_build, same model / loss / label rule as the
training code). Adam with a jitted step records only the loss
(make_logging_trainer also re-evaluates the loss and logs all 397 weights every
step, which costs ~4x more here; lax.scan is ~3x slower than a Python loop over a
jitted step on this CPU). Loss curve entry k = full-train loss after k updates.
Run from the project root:
    python -m kinkchain.utkface_feasibility
    python -m kinkchain.utkface_feasibility --lrs 1e-3 1e-2 --seeds 0 1 --taus 40
    python -m kinkchain.utkface_feasibility --channels 24 --taus 40     # wider net
    python -m kinkchain.utkface_feasibility --taus 40 --clip-norm 1.0   # gradient clipping
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .mam_weights_reparam_alt import WMAMConfig, _build
from .reparam import reparam_alt

BINS = ((0, 2), (3, 5), (6, 10), (11, 1000))


def bin_name(lo, hi):
    return f"{lo}-{hi}" if hi < 1000 else f">{lo - 1}"


def make_runner(B, lr, afactor, max_steps, clip_norm=None, clip_value=None):
    """Jitted single Adam step (full batch, identity reparam when kink_fraction=0).
    A Python loop over this is ~3x faster here than the same step inside
    lax.scan / while_loop (54 vs 176 ms), so we do not scan.
    clip_norm: clip the global gradient norm to this value before Adam.
    clip_value: clip every gradient element to [-c, c] before Adam (this is what
    `--grad-clip` does in the training code). Both None -> plain Adam, unchanged."""
    def logits(th, X):
        return B["model"].apply({"params": reparam_alt(th, B["knots"], B["scales"], B["mask"])}, X)

    def loss_fn(th, tau):
        lab = (B["atr"] >= tau).astype(jnp.float32)
        return optax.sigmoid_binary_cross_entropy(logits(th, B["Xtr"]), lab).mean()

    opt = optax.adam(optax.cosine_decay_schedule(lr, max_steps, afactor))
    pre = ([optax.clip_by_global_norm(clip_norm)] if clip_norm else []) + \
          ([optax.clip(clip_value)] if clip_value else [])
    if pre:
        opt = optax.chain(*pre, opt)

    @jax.jit
    def step(th, st, tau):
        l, g = jax.value_and_grad(loss_fn)(th, tau)      # l = loss at th, before the update
        up, st = opt.update(g, st, th)
        return optax.apply_updates(th, up), st, l

    return opt, step, jax.jit(loss_fn), logits


def run_one(B, runner, tau, seed, max_steps, tol, chunk, progress=0):
    """Train from a random init at one tau; every `chunk` steps check whether the
    loss has reached tol and stop if so. progress > 0 prints a line every that many
    steps (a multiple of chunk). Returns (loss_curve, theta_final, logits)."""
    opt, step, final_loss, logits = runner
    th = B["model"].init(jax.random.PRNGKey(seed), B["Xtr"][:1])["params"]
    st, ls, done, n_spikes, t0 = opt.init(th), [], 0, 0, time.time()
    while done < max_steps:
        buf = []
        for _ in range(chunk):
            th, st, l = step(th, st, float(tau))
            buf.append(l)
        ls.append(np.asarray(jnp.stack(buf)))
        if progress:
            c = ls[-1] if len(ls) == 1 else np.concatenate([ls[-2][-1:], ls[-1]])
            n_spikes += int((c[1:] > 2 * np.maximum(c[:-1], 1e-300)).sum())
        done += chunk
        if progress and done % progress == 0:
            print(f"  [tau {tau:g}] step {done:7d}  loss {ls[-1][-1]:.3e}  spikes(>2x) so far {n_spikes}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if (ls[-1] <= tol).any():
            break
    curve = np.concatenate(ls + [np.asarray(final_loss(th, float(tau)))[None]])
    return curve, th, logits


def evaluate(B, th, logits, tau):
    ytr = np.asarray(B["atr"]) >= tau
    yte = np.asarray(B["ate"]) >= tau
    ptr = np.asarray(logits(th, B["Xtr"])) > 0
    pte = np.asarray(logits(th, B["Xte"])) > 0
    dist = np.abs(np.asarray(B["ate"]) - tau)
    res = dict(train_acc=float((ptr == ytr).mean()), test_acc=float((pte == yte).mean()),
               majority_test_acc=float(max(yte.mean(), 1 - yte.mean())))
    for lo, hi in BINS:
        m = (dist >= lo) & (dist <= hi)
        res[f"n_{bin_name(lo, hi)}"] = int(m.sum())
        res[f"acc_{bin_name(lo, hi)}"] = float((pte[m] == yte[m]).mean()) if m.any() else float("nan")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--taus", type=float, nargs="+", default=[38, 39, 40, 41, 42, 43])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-3])
    p.add_argument("--afactor", type=float, default=1.0,
                   help="cosine decay of the lr to lr*afactor over max-steps (1 = constant)")
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--tol", type=float, default=1e-6,
                   help="stop (at the end of the chunk) once the full-train loss <= tol")
    p.add_argument("--chunk", type=int, default=100, help="check for early stop (loss <= tol) every this many steps")
    p.add_argument("--progress", type=int, default=0,
                   help="print step / loss / spike count every this many steps (default 0: off)")
    p.add_argument("--clip-norm", type=float, default=None,
                   help="clip the GLOBAL gradient norm to this value before Adam (default: off)")
    p.add_argument("--clip-value", type=float, default=None,
                   help="clip every gradient element to [-c, c] before Adam, as --grad-clip does in "
                        "the training code (default: off)")
    p.add_argument("--channels", type=int, default=6)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default="runs/utkface_feasibility.npz")
    a = p.parse_args()

    if a.max_steps % a.chunk:
        p.error("--max-steps must be a multiple of --chunk")
    cfg = WMAMConfig(img=64, data=a.data or "data/utkface_100_150_64.npz", tau_min=38.0, tau_max=43.0,
                     increment=1.0, near_band=6.0, kink_fraction=0.0, channels=a.channels, batch_size=0)
    B = _build(cfg)
    n_par = sum(x.size for x in jax.tree_util.tree_leaves(B["params0"]))
    print(f"data={cfg.data}  channels={cfg.channels}  params={n_par}  kinked={int(B['kink'].sum())}  "
          f"adam max_steps={a.max_steps} tol={a.tol} afactor={a.afactor} "
          f"clip_norm={a.clip_norm} clip_value={a.clip_value}")

    runners = {lr: make_runner(B, lr, a.afactor, a.max_steps, a.clip_norm, a.clip_value) for lr in a.lrs}
    curves, rows = {}, []
    hdr = ("tau  seed    lr  steps*  <tol  final_loss   min_loss  train_acc test_acc "
           "(majority)  | test acc by |age-tau|: " + "  ".join(bin_name(*b) for b in BINS))
    print(hdr)
    print("(steps* = first step with loss <= tol, or the number of steps run if never reached)")
    for lr in a.lrs:
        for seed in a.seeds:
            for tau in a.taus:
                t0 = time.time()
                curve, th, logits = run_one(B, runners[lr], tau, seed, a.max_steps, a.tol, a.chunk, a.progress)
                r = evaluate(B, th, logits, tau)
                hit = bool((curve <= a.tol).any())
                first = int(np.argmax(curve <= a.tol)) if hit else len(curve) - 1
                r.update(tau=tau, seed=seed, lr=lr, steps=first, reached_tol=hit,
                         final_loss=float(curve[-1]), min_loss=float(curve.min()),
                         clip_norm=a.clip_norm or 0.0, clip_value=a.clip_value or 0.0)   # 0 = off
                rows.append(r)
                curves[f"loss_tau{tau:g}_seed{seed}_lr{lr:g}"] = curve
                print(f"{tau:3g}  {seed:4d} {lr:6.0e} {r['steps']:6d}  {'Y' if hit else 'n'}   "
                      f"{r['final_loss']:10.3e} {r['min_loss']:10.3e}   {r['train_acc']:.2f}     "
                      f"{r['test_acc']:.2f}   ({r['majority_test_acc']:.2f})   | "
                      + "  ".join(f"{r[f'acc_{bin_name(*b)}']:.2f}(n={r[f'n_{bin_name(*b)}']})" for b in BINS)
                      + f"   [{time.time() - t0:.0f}s]", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    keys = [k for k in rows[0] if k != "reached_tol"]
    np.savez(a.out, **{k: np.array([r[k] for r in rows]) for k in keys},
             reached_tol=np.array([r["reached_tol"] for r in rows]),
             **curves)
    print(f"saved summary + loss curves -> {a.out}")


if __name__ == "__main__":
    main()
