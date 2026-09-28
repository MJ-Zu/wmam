"""Weight-adapting MAM harness -- a miniature `kinkchain`.

Training mirrors the adaptation made in mam.py:
  * BULK cyclic training uses a selectable optimizer (default RMSProp WITHOUT
    momentum) with a cosine-decayed learning rate, so a single un-tuned base rate
    both converges (large early, to track the moving target) and settles (small
    late). RMSProp-no-momentum keeps the step close to steepest descent -- far
    less off-gradient drift than Adam, which the paper (Sec IV C) warns can
    substitute for the kink-binding we are trying to isolate.
  * A MAM-approach gate ends the bulk phase early once (1) every target in a
    cycle has converged (max loss < converge_tol) AND (2) dD has plateaued.
  * A REFINE phase of plain fixed-lr GD follows, for a clean final state.

Two independent noise axes (do not conflate):
  * TASK-HARDENING (property_noise, small n_train): a DATA property that makes
    the solution underdetermined -> gives the held-out reversibility readout its
    dynamic range. Training stays deterministic. HELPS.
  * OPTIMIZATION NOISE (batch_size>0, augment): a TRAINING-DYNAMICS property
    (Keim-Nagel thermal noise), applied only during bulk cycling. ERODES memory.

Readout is behavioral (robust to weight-space gauge symmetry): fixed-budget
retrain loss on TRAIN, and return-point reversibility on HELD-OUT (overall and
near the tau_max boundary). Deviation from mam.py: the readout uses a MODERATE
plain-GD lr, not a tiny one -- a classification loss needs enough step to
actually re-fit labels / traverse to tau_read; a tiny lr cannot and the signal
vanishes. The refine phase provides the clean fixed-lr settling instead.

Run from the project root:
    python -m kinkchain.mam_weights_logging                 # bulk cyclic training
    python -m kinkchain.mam_weights_logging --refine --load runs/wmam_cyclic.npz   # refine cyclic training
    python -m kinkchain.mam_weights_logging --readout       # readout
    python -m kinkchain.mam_weights_logging --readout --load runs/wmam_refine_cyclic.npz
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

from .reparam import build_kink_params, reparam_alt, kink_stats
from .mam import theta_perp, _rms, _dD_plateaued


@dataclass
class WMAMConfig:
    img: int = 16               # image size 16x16
    n_train: int = 100           # small -> underdetermined (task-hardening)
    n_test: int = 150
    channels: int = 6		# number of channels in CNN
    kink_fraction: float = 0.0
    s_low: float = 1.0          # amplitude of kinks s, theta = theta + s|theta-c|, 1.5, 3.0
    s_high: float = 1.0
    property_noise: float = 0.0  # noise in the image's encoding of a (hardening)
    data: str = "data/disk_data_100_150.npz"
#    data: str=None
    tau_min: float = 0.5
    tau_max: float = 0.55
    increment: float = 0.01
    near_band: float = 0.12      # |a - tau_max| < near_band = "near boundary"
    # --- bulk phase (RMSProp-no-momentum + cosine-decayed lr) ---
    bulk_optimizer: str = "adam"   # "rmsprop" | "adam" | "gd"
    lr: float = 1e-1                  # 1e-2bulk base lr
    bulk_afactor: float = 0.9         # 0.3 cosine decay: lr -> lr*afactor over max_steps
    n_cycles: int = 20                # max bulk cycles (stops early on MAM-approach)
    max_steps: int = 500              # per-target step cap
    tol: float = 1e-11                # per-target convergence tol (classification) 1e-4
    # --- MAM-approach detection ---
    converge_tol: float = 1e-11        # cycle "converged" if every target's loss < this
    dD_patience: int = 3
    dD_plateau_rel: float = 0.01
    # --- refine phase (plain fixed-lr GD) ---
    refine_cycles: int = 100
    refine_lr: float = 1e-4
    refine_max_steps: int = 1000
    refine_tol: float = 1e-10
    # --- readout (plain GD, MODERATE lr so labels can be re-fit) ---
    readout_lr: float = 1e-4
    
    readout_max_steps: int = 1000
    budget_steps: int = 500
    readout_tol: float = 1e-10
    # --- optimization noise (off by default; applied only during bulk) ---
    batch_size: int = 0
    augment: bool = False
    seed: int = 0

    def as_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------
# Synthetic data: disk radius encodes a (noisily if property_noise>0); label = (true a >= tau)
# ----------------------------------------------------------------------------
def make_data(cfg: WMAMConfig):
    rng = np.random.default_rng(cfg.seed + 7)
    N = cfg.n_train + cfg.n_test
    H = cfg.img
    a = rng.uniform(0.0, 1.0, N)
    a_img = np.clip(a + rng.normal(0.0, cfg.property_noise, N), 0.0, 1.0)
    yy, xx = np.mgrid[0:H, 0:H]
    X = np.zeros((N, H, H, 1), np.float32)
    for i in range(N):
        cy, cx = rng.uniform(0.35, 0.65, 2) * H
        R = (0.12 + 0.45 * a_img[i]) * H
        disk = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= R).astype(np.float32)
        X[i, :, :, 0] = np.clip(disk + rng.normal(0, 0.1, (H, H)), 0.0, 1.5)
    X = (X - X.mean()) / (X.std() + 1e-6)
    ntr = cfg.n_train
    return (jnp.asarray(X[:ntr]), jnp.asarray(a[:ntr].astype(np.float32)),
            jnp.asarray(X[ntr:]), jnp.asarray(a[ntr:].astype(np.float32)))


class TinyCNN(nn.Module):
    channels: int = 6
    act: callable = nn.gelu
    #act: callable = nn.relu ## ReLu is continue but has a gradient discontinuity at x=0

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        x = self.act(x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME")(x)
        x = self.act(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(1)(x)
        return x[..., 0]


## randomly flip or tuning brightness
def _augment(Xb, key):
    k1, k2 = jax.random.split(key)
    b = Xb.shape[0]
    flip = jax.random.bernoulli(k1, 0.5, (b,))
    Xb = jnp.where(flip[:, None, None, None], Xb[:, :, ::-1, :], Xb)
    bright = jax.random.uniform(k2, (b, 1, 1, 1), minval=-0.3, maxval=0.3)
    return Xb + bright



def make_budget(model, knots, scales, mask, Xtr, atr, steps, lr, grad_clip=None):
    """Fixed-budget full-batch plain GD; returns residual full-batch loss."""
    def logits(th, X):
        return model.apply({"params": reparam_alt(th, knots, scales, mask)}, X)
    def full_loss(th, tau):
        lab = (atr >= tau).astype(jnp.float32)
        return optax.sigmoid_binary_cross_entropy(logits(th, Xtr), lab).mean()
    gfull = jax.grad(full_loss)
#    opt = optax.sgd(lr)
    opt = (optax.chain(optax.clip(grad_clip), optax.sgd(lr, momentum=None))
           if grad_clip is not None else optax.sgd(lr, momentum=None))

    @jax.jit
    def budget(theta, tau):
        st = opt.init(theta)
        def step(i, c):
            th, s = c
            up, s = opt.update(gfull(th, tau), s, th)
            return optax.apply_updates(th, up), s
        p, _ = jax.lax.fori_loop(0, steps, step, (theta, st))
        return full_loss(p, tau)
    return budget


# ----------------------------------------------------------------------------
# Per-step LOGGING trainer: identical optimizer/loss/body to make_trainer, but
# records the flattened parameter vector and the full-batch loss at EVERY step
# into static (max_steps+1, n) buffers. The while_loop still early-exits at
# convergence, so no steps are wasted; `_truncate` drops the untouched tail.
# ----------------------------------------------------------------------------
def make_logging_trainer(model, knots, scales, mask, Xtr, atr, Xte, ate, *, max_steps, tol,
                         opt_name, lr, afactor, batch=0, grad_clip=None, augment=False):
    n_data = Xtr.shape[0]

    def logits(th, X):
        return model.apply({"params": reparam_alt(th, knots, scales, mask)}, X)

    def full_loss(th, tau):
        lab = (atr >= tau).astype(jnp.float32)
        return optax.sigmoid_binary_cross_entropy(logits(th, Xtr), lab).mean()

    def batch_loss(th, tau, X, a):
        lab = (a >= tau).astype(jnp.float32)
        return optax.sigmoid_binary_cross_entropy(logits(th, X), lab).mean()
    
    def full_loss_valid(th, tau):
        lab = (ate >= tau).astype(jnp.float32)
        return optax.sigmoid_binary_cross_entropy(logits(th, Xte), lab).mean()

    gbatch = jax.grad(batch_loss)
    sched = optax.cosine_decay_schedule(lr, max_steps, afactor)
    if grad_clip is None:
       opt = {"gd": optax.sgd(sched, momentum=None),
              "rmsprop": optax.rmsprop(sched, momentum=None),
              "adam": optax.adam(sched)}[opt_name]
    else:
       opt = {"gd": optax.chain(optax.clip(grad_clip), optax.sgd(sched, momentum=None)),
              "rmsprop": optax.chain(optax.clip(grad_clip), optax.rmsprop(sched, momentum=None)),
              "adam": optax.chain(optax.clip(grad_clip), optax.adam(sched))}[opt_name]

    @jax.jit
    def train_log(theta, tau, key):
        flat0, _ = ravel_pytree(theta)
        n = flat0.shape[0]
        rows = max_steps + 1
        st = opt.init(theta)
        l0 = full_loss(theta, tau)
        bt0 = jnp.zeros((rows, n), flat0.dtype).at[0].set(flat0)   # row 0 = init
        bl0 = jnp.zeros((rows,), l0.dtype).at[0].set(l0)

        def cond(c):
            _, _, i, l, _, _, _ = c
            return (i < max_steps) & (l > tol)

        def body(c):
            th, s, i, _, k, bt, bl = c
            k, ks, ka = jax.random.split(k, 3)
            if batch and batch < n_data:
                idx = jax.random.choice(ks, n_data, (batch,), replace=False)
                Xb, ab = Xtr[idx], atr[idx]
            else:
                Xb, ab = Xtr, atr
            if augment:
                Xb = _augment(Xb, ka)
            up, s = opt.update(gbatch(th, tau, Xb, ab), s, th)
            th = optax.apply_updates(th, up)
            l = full_loss(th, tau)
            i = i + 1
            bt = bt.at[i].set(ravel_pytree(th)[0])   # flatten with the same order
            bl = bl.at[i].set(l)
            return th, s, i, l, k, bt, bl

        th, s, i, l, k, bt, bl = jax.lax.while_loop(
            cond, body, (theta, st, 0, l0, key, bt0, bl0))
        # valid rows are 0..i; slice [:i+1] to drop the untouched tail.
        return th, i, l, bt, bl

    return train_log, full_loss, full_loss_valid, logits


def _truncate(bt, bl, n_steps, stride=1):
    """Host-side: keep rows 0..n_steps (the real trajectory), optionally thinned."""
    k = int(n_steps) + 1
    bt = np.asarray(bt[:k]); bl = np.asarray(bl[:k])
    if stride > 1 and k > 2:
        keep = list(range(0, k, stride))
        if keep[-1] != k - 1:
            keep.append(k - 1)               # always keep the final point
        bt, bl = bt[keep], bl[keep]
    return bt, bl


def _triangle(cfg):
    tmax, tmin, inc = cfg.tau_max, cfg.tau_min, cfg.increment
    down = list(np.round(np.arange(tmax - inc, tmin - 1e-9, -inc), 6))
    up = list(np.round(np.arange(tmin + inc, tmax + 1e-9, inc), 6))
    return down, up



def _contrasts(r):
    ins = r["inside"]
    def con(a):
        a = a[~np.isnan(a)] if np.isnan(a).any() else a
        return float(a[~ins].mean() / max(a[ins].mean(), 1e-30))
    return (con(r["mam_fixed_budget_loss"]), con(r["mam_rev_prob"]),
            con(r["mam_rev_prob_near"]))


def sweep_batch(batch_sizes, seeds, out_path="runs/wmam_batch_sweep.npz", **base):
    """Graded gradient-noise study: hold everything fixed on the hardened task,
    sweep only the mini-batch size (batch_size=0 == full-batch deterministic
    baseline; smaller batch == noisier gradient). Multiple seeds per point.
    Saves 2D arrays [n_batch, n_seeds] of the memory contrasts."""
    B, S = len(batch_sizes), len(seeds)
    retrain = np.zeros((B, S)); rev_all = np.zeros((B, S)); rev_near = np.zeros((B, S))
    for i, bs in enumerate(batch_sizes):
        for j, sd in enumerate(seeds):
            cfg = WMAMConfig(batch_size=int(bs), augment=False, seed=int(sd), **base)
            r = run_wmam(cfg, verbose=False, do_cold=False)
            retrain[i, j], rev_all[i, j], rev_near[i, j] = _contrasts(r)
        print(f"  batch={int(bs):<4} ({'full' if bs == 0 else 'noisy'})  "
              f"retrain median={np.median(retrain[i]):5.1f}x  "
              f"rev_near median={np.median(rev_near[i]):4.1f}x")
    np.savez(out_path, batch_sizes=np.asarray(batch_sizes, float),
             seeds=np.asarray(seeds), retrain=retrain, rev_all=rev_all,
             rev_near=rev_near)
    print(f"saved {out_path}")
    return dict(batch_sizes=np.asarray(batch_sizes, float), retrain=retrain,
                rev_all=rev_all, rev_near=rev_near)


# ============================================================================
# Per-step-logged cyclic training and readout (share the same building blocks
# as run_wmam; the trainer is make_logging_trainer, so the dynamics are the
# SAME as run_wmam -- only the full path is recorded).
# ============================================================================
def _build(cfg):
    """Data + model + kink config + flatten helpers (mirrors run_wmam setup)."""
    if cfg.data is None:
        Xtr, atr, Xte, ate = make_data(cfg)
    elif isinstance(cfg.data, str):
        from .disk_data import load_disk_data
        Xtr, atr, Xte, ate = load_disk_data(cfg.data)
    else:
        raise ValueError("no dataset")
    near = np.abs(np.asarray(ate) - cfg.tau_max) < cfg.near_band
    model = TinyCNN(channels=cfg.channels)
    params0 = model.init(jax.random.PRNGKey(cfg.seed), Xtr[:1])["params"]
    knots, scales, mask = build_kink_params(
        params0, jax.random.PRNGKey(cfg.seed + 99),
        fraction=cfg.kink_fraction, s_low=cfg.s_low, s_high=cfg.s_high,
        knot_mode="zero")
    _, unravel = ravel_pytree(params0)
    kink = np.asarray(ravel_pytree(mask)[0]) != 0
    return dict(Xtr=Xtr, atr=atr, Xte=Xte, ate=ate, near=near, model=model,
                params0=params0, knots=knots, scales=scales, mask=mask,
                unravel=unravel, kink=kink,
                flat=lambda p: np.asarray(ravel_pytree(p)[0]))

def save_setup(B, cfg, path):
    """Freeze the reproducibility-critical setup to a .npz: the dataset
    (Xtr/atr/Xte/ate), the derived near-boundary mask and kink mask, and the
    flattened params0/knots/scales/mask pytrees, plus cfg. Loading this makes
    the SETUP (data, init, kink config) byte-identical across runs regardless
    of backend RNG. It does NOT freeze training-time backend nondeterminism
    (e.g. GPU float ops) -- for that also set deterministic ops + x64."""
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    f = B["flat"]
    np.savez_compressed(
        path,
        Xtr=np.asarray(B["Xtr"]), atr=np.asarray(B["atr"]),
        Xte=np.asarray(B["Xte"]), ate=np.asarray(B["ate"]),
        near=np.asarray(B["near"]), kink=np.asarray(B["kink"]),
        params0=f(B["params0"]), knots=f(B["knots"]),
        scales=f(B["scales"]), mask=f(B["mask"]),
        _cfg=np.array(json.dumps(cfg.as_dict())))
    return path
 
 
def load_setup(path):
    """Rebuild a B dict identical to _build() from a save_setup() file, and the
    cfg it was built with. The param-pytree STRUCTURE is regenerated from cfg
    (deterministic: shapes don't depend on RNG) and used to unflatten the saved
    vectors, so no functions/treedefs need to be pickled. Returns (B, cfg)."""
    z = np.load(path, allow_pickle=False)
    cfg = WMAMConfig(**json.loads(str(z["_cfg"])))
    Xtr = z["Xtr"]
    model = TinyCNN(channels=cfg.channels)
    template = model.init(jax.random.PRNGKey(cfg.seed), Xtr[:1])["params"]
    _, unravel = ravel_pytree(template)          # structure only; values unused
    B = dict(
        Xtr=Xtr, atr=z["atr"], Xte=z["Xte"], ate=z["ate"],
        near=z["near"], kink=z["kink"], model=model,
        params0=unravel(jnp.asarray(z["params0"])),
        knots=unravel(jnp.asarray(z["knots"])),
        scales=unravel(jnp.asarray(z["scales"])),
        mask=unravel(jnp.asarray(z["mask"])),
        unravel=unravel,
        flat=lambda p: np.asarray(ravel_pytree(p)[0]))
    return B, cfg


def _run_cycle_logged(train_log, theta, down, up, key, rec, cyc, stride):
    max_loss = 0.0
    for t in down:
        key, kk = jax.random.split(key)
        theta, ns, l, bt, bl = train_log(theta, float(t), kk)
        rec(bt, bl, ns, cyc, float(t), "down"); max_loss = max(max_loss, float(l))
    theta_min = theta
    for t in up:
        key, kk = jax.random.split(key)
        theta, ns, l, bt, bl = train_log(theta, float(t), kk)
        rec(bt, bl, ns, cyc, float(t), "up"); max_loss = max(max_loss, float(l))
    return theta, theta_min, max_loss, key

def count_knot_crossings(theta_traj, knots_flat, kink_mask):
    """How many SPECIAL parameters cross their own knot at least once across
    a logged (n_steps, n_params) trajectory.

    A crossing is a sign change of (theta - c) between two consecutive
    logged steps: the parameter was on one side of its kink and ended up on
    the other at some point. Non-special parameters (kink_mask False) are
    ignored. A parameter sitting exactly on its knot (diff == 0) contributes
    sign 0; a transition into/out of exactly 0 counts as a change too, since
    np.sign(0) == 0 differs from +-1.

    Args:
        theta_traj: (n_steps, n_params) array -- one row per logged step.
        knots_flat: (n_params,) flattened knot values c (same layout as
            theta_traj's columns, e.g. from B["flat"](B["knots"])).
        kink_mask: (n_params,) boolean, True where the parameter is special
            (e.g. B["kink"]).

    Returns:
        (n_crossed, crossed_mask): the count of distinct special parameters
        that crossed at least once, and a boolean (n_params,) array marking
        exactly which ones (False everywhere for non-special parameters).
    """
    theta_traj = np.asarray(theta_traj)
    knots_flat = np.asarray(knots_flat)
    kink_mask = np.asarray(kink_mask)

    special_idx = np.where(kink_mask)[0]
    diff = theta_traj[:, special_idx] - knots_flat[special_idx]   # (n_steps, n_special)
    sign = np.sign(diff)
    changed = np.any(sign[1:] != sign[:-1], axis=0) if len(sign) > 1 else np.zeros(
        special_idx.shape[0], dtype=bool)

    crossed_mask = np.zeros(kink_mask.shape, dtype=bool)
    crossed_mask[special_idx] = changed
    return int(changed.sum()), crossed_mask

def cyclic_train_logged(cfg=None, verbose=True, out_path="runs/wmam_cyclic.npz",
                        setup=None, grad_clip=None,
                        stride=1):
    """Cold -> bulk cycles -> refine cycles, logging theta and loss every step.

    Returns a dict of flat, step-aligned arrays:
      cycle (S,) int  (0 = cold training to tau_max) | tau (S,) | leg (S,)
      loss  (S,)      | theta (S, n_params)  [flattened weights]
    plus kink (n_params,) bool and theta_cold/theta_max/theta_min (flattened).
    Saves to out_path (recallable by readout). Also returns the pytree
    endpoints under _theta_max_tree / _theta_min_tree for an in-process readout.
    """
    cfg = cfg or WMAMConfig()
    B = setup if setup is not None else _build(cfg)
    common = dict(model=B["model"], knots=B["knots"], scales=B["scales"], mask=B["mask"],
                  Xtr=B["Xtr"], atr=B["atr"], Xte=B["Xte"], ate=B["ate"])
    bulk_log, _, full_loss_valid, _ = make_logging_trainer(
        **common, max_steps=cfg.max_steps, tol=cfg.tol,
        opt_name=cfg.bulk_optimizer, lr=cfg.lr, afactor=cfg.bulk_afactor,
        batch=cfg.batch_size, grad_clip=grad_clip, augment=cfg.augment)
    cold_log, _, _, _ = make_logging_trainer(
        **common, max_steps=cfg.refine_max_steps, tol=cfg.refine_tol,
        opt_name="gd", lr=cfg.refine_lr, afactor=1.0, grad_clip=grad_clip)

    cyc_c, tau_c, leg_c, loss_c, th_rows = [], [], [], [], []

    def rec(bt, bl, ns, cyc, tau, leg):
        bt, bl = _truncate(bt, bl, ns, stride)
        k = len(bl)
        cyc_c.extend([cyc] * k); tau_c.extend([tau] * k)
        leg_c.extend([leg] * k); loss_c.extend(bl.tolist()); th_rows.append(bt)
    
    flatf = B["flat"]
    knots_flat = flatf(B["knots"])
    kink_mask = B["kink"]
    dummy = jax.random.PRNGKey(0)
    down, up = _triangle(cfg)
    key = jax.random.PRNGKey(cfg.seed + 1234)

    theta, ns, lc, bt, bl = cold_log(B["params0"], cfg.tau_max, dummy)
    rec(bt, bl, ns, 0, cfg.tau_max, "init")
    theta_cold = theta
    if verbose:
        print(f"loss_cold={float(lc):.6e}")

    dD, prev, theta_min, mam_cycle, count = [], theta, theta, -1, 0
    knot_crossings = []
    for cyc in range(1, cfg.n_cycles + 1):
        rows_before = len(th_rows)
        theta, theta_min, ml, key = _run_cycle_logged(
            bulk_log, theta, down, up, key, rec, cyc, stride)
        dD.append(_rms(jnp.asarray(flatf(theta) - flatf(prev)))); prev = theta
        
        cycle_theta = np.concatenate(th_rows[rows_before:], axis=0)
        n_crossed, crossed_mask = count_knot_crossings(cycle_theta, knots_flat, kink_mask)
        knot_crossings.append(n_crossed)

        converged = ml < cfg.converge_tol
        done = converged and (n_crossed == 0)
        #plateaued = _dD_plateaued(dD, cfg.dD_patience, cfg.dD_plateau_rel)
        vl = full_loss_valid(theta, up[-1])
        if verbose and (cyc == 1 or cyc % 3 == 0):
            print(f"  bulk {cyc:3d}  dD={dD[-1]:.2e}  max_loss={ml:.2e}  "
                  f"converged={converged}  knot_crossings={n_crossed}  "
                  f"valid_loss={vl:.2e}")
#       if converged and plateaued:
#           mam_cycle = cyc; count += 1
#           if count > cfg.dD_patience:
#               if verbose:
#                   print(f"  -> MAM approached at bulk cycle {cyc}")
#               break
#       else:
#           count = 0
        if done:
            mam_cycle = cyc
            if verbose:
                print(f"  -> MAM approached at bulk cycle {cyc}")
            break
          
    n_bulk = len(dD)

#   for r in range(cfg.refine_cycles):
#       theta, theta_min, ml, key = _run_cycle_logged(
#           refine_log, theta, down, up, key, rec, n_bulk + 1 + r, stride)
#       dD.append(_rms(jnp.asarray(flatf(theta) - flatf(prev)))); prev = theta
#       vl = full_loss_valid(theta, up[-1])
#       if verbose:
#           print(f"  refine {r + 1:2d}  dD={dD[-1]:.2e}  max_loss={ml:.2e}  "
#                 f"valid_loss={vl:.2e}")
    theta_max = theta

    log = dict(
        kind="cyclic", seed=cfg.seed,
        cycle=np.array(cyc_c), tau=np.array(tau_c),
        leg=np.array(leg_c, dtype="<U8"), loss=np.array(loss_c),
        theta=np.concatenate(th_rows, axis=0), kink=B["kink"],
        theta_cold=flatf(theta_cold), theta_max=flatf(theta_max),
        theta_min=flatf(theta_min), dD=np.array(dD),
        knot_crossings=np.array(knot_crossings),
        n_bulk=n_bulk, mam_cycle=mam_cycle)
    if out_path:
        save_log(log, out_path, meta=dict(cfg=cfg.as_dict()))
        if verbose:
            print(f"[cyclic] {log['theta'].shape[0]} steps logged -> {out_path}")
    log["_theta_max_tree"] = theta_max      # pytrees for in-process readout
    log["_theta_min_tree"] = theta_min
    return log

def refine_cyclic_train_logged(cfg=None, load=None, verbose=True, out_path="runs/wmam_refine_cyclic.npz",
                        setup=None, further=False, grad_clip=None,
                        stride=1):
    """refine cycles, logging theta and loss every step.

    Returns a dict of flat, step-aligned arrays:
      cycle (S,) int  (0 = cold training to tau_max) | tau (S,) | leg (S,)
      loss  (S,)      | theta (S, n_params)  [flattened weights]
    plus kink (n_params,) bool and theta_cold/theta_max/theta_min (flattened).
    Saves to out_path (recallable by readout). Also returns the pytree
    endpoints under _theta_max_tree / _theta_min_tree for an in-process readout.
    """
    cfg = cfg or WMAMConfig()
    B = setup if setup is not None else _build(cfg)
    flatf = B["flat"]
    common = dict(model=B["model"], knots=B["knots"], scales=B["scales"], mask=B["mask"],
                  Xtr=B["Xtr"], atr=B["atr"], Xte=B["Xte"], ate=B["ate"])
    if further:
      refine_log, _, full_loss_valid, _ = make_logging_trainer(
        **common, max_steps=cfg.max_steps, tol=cfg.tol,
        opt_name=cfg.bulk_optimizer, lr=cfg.lr, afactor=cfg.bulk_afactor,
        grad_clip=grad_clip)
      print("further bulk cyclic training")
    else:
      refine_log, _, full_loss_valid, _ = make_logging_trainer(
        **common, max_steps=cfg.refine_max_steps, tol=cfg.refine_tol,
        opt_name="gd", lr=cfg.refine_lr, afactor=1.0, grad_clip=grad_clip)
      print(cfg.kink_fraction, cfg.refine_lr, cfg.refine_tol, cfg.refine_max_steps)


    if load:
        prev = load_log(load)
        theta_max = B["unravel"](jnp.asarray(prev["theta_max"]))
        theta_min = B["unravel"](jnp.asarray(prev["theta_min"]))
        if verbose:
            print(f"[refine] recalled theta_max/theta_min from {load}")
    else:
        if verbose:
            print("[refine] no --load; running a cyclic pass first for theta_max")
        clog = cyclic_train_logged(cfg, verbose=verbose, out_path=None, stride=stride)
        theta_max, theta_min = clog["_theta_max_tree"], clog["_theta_min_tree"]

    cyc_c, tau_c, leg_c, loss_c, th_rows = [], [], [], [], []

    def rec(bt, bl, ns, cyc, tau, leg):
        bt, bl = _truncate(bt, bl, ns, stride)
        k = len(bl)
        cyc_c.extend([cyc] * k); tau_c.extend([tau] * k)
        leg_c.extend([leg] * k); loss_c.extend(bl.tolist()); th_rows.append(bt)
    
    flatf = B["flat"]
    knots_flat = flatf(B["knots"])
    kink_mask = B["kink"]
    dummy = jax.random.PRNGKey(0)
    down, up = _triangle(cfg)
    key = jax.random.PRNGKey(cfg.seed + 12345)

    dD, prev, theta_min, mam_cycle, count = [], theta_max, theta_min, -1, 0
    theta = theta_max
    knot_crossings = []
    for r in range(cfg.refine_cycles):
        rows_before = len(th_rows)
        theta, theta_min, ml, key = _run_cycle_logged(
            refine_log, theta, down, up, key, rec, 1 + r, stride)
        dD.append(_rms(jnp.asarray(flatf(theta) - flatf(prev)))); prev = theta
        
        cycle_theta = np.concatenate(th_rows[rows_before:], axis=0)
        n_crossed, crossed_mask = count_knot_crossings(cycle_theta, knots_flat, kink_mask)
        knot_crossings.append(n_crossed)

        converged = ml < cfg.refine_tol
        done = converged and (n_crossed == 0)
        vl = full_loss_valid(theta, up[-1])
        if verbose:
            print(f"  refine {r + 1:2d}  dD={dD[-1]:.2e}  max_loss={ml:.2e}  "
                  f"  knot_crossing={n_crossed}  valid_loss={vl:.2e}")
        if done :
            mam_cycle = r; count += 1
   #      if count > cfg.dD_patience:
            if verbose:
                print(f"  -> MAM approached at refine cycle {r}")
                break
        else:
            count = 0
    n_refine = len(dD)
    theta_max = theta

    kind = "further_cyclic" if further else "refine" 
    log = dict(
        kind=kind, seed=cfg.seed,
        cycle=np.array(cyc_c), tau=np.array(tau_c),
        leg=np.array(leg_c, dtype="<U8"), loss=np.array(loss_c),
        theta=np.concatenate(th_rows, axis=0), kink=B["kink"],
        theta_max=flatf(theta_max), theta_min=flatf(theta_min), 
        dD=np.array(dD),
        knot_crossings=np.array(knot_crossings),
        n_refine=n_refine, mam_cycle=mam_cycle)
    if out_path:
        save_log(log, out_path, meta=dict(cfg=cfg.as_dict()))
        if verbose:
            print(f"[cyclic] {log['theta'].shape[0]} steps logged -> {out_path}")
    log["_theta_max_tree"] = theta_max      # pytrees for in-process readout
    log["_theta_min_tree"] = theta_min
    return log

def readout_logged(cfg=None, load=None, verbose=True, do_cold=None, setup=None,
                   out_path="runs/wmam_readout.npz", grad_clip=None, stride=1):
    """Behavioral readout (same observables as run_wmam's do_readout), logging
    every step of each retrain leg. theta_max/theta_min come from a recalled
    cyclic log (`load=...`) or from a fresh cyclic pass if none is given."""
    cfg = cfg or WMAMConfig()
    B = setup if setup is not None else _build(cfg)
    flatf = B["flat"]
    common = dict(model=B["model"], knots=B["knots"], scales=B["scales"], mask=B["mask"],
                  Xtr=B["Xtr"], atr=B["atr"], Xte=B["Xte"], ate=B["ate"])
    ro_log, _, _, logits = make_logging_trainer(
        **common, 
        max_steps=cfg.readout_max_steps, 
        tol=cfg.readout_tol,
        opt_name="gd", lr=cfg.refine_lr, afactor=1.0, grad_clip=grad_clip)
    budget = make_budget(B["model"], B["knots"], B["scales"], B["mask"],
                         B["Xtr"], B["atr"], cfg.budget_steps, cfg.refine_lr)
    print(cfg.refine_lr)

    def probs_test(p):
        return np.asarray(jax.nn.sigmoid(logits(p, B["Xte"])))

    if load:
        prev = load_log(load)
        theta_max = B["unravel"](jnp.asarray(prev["theta_max"]))
        theta_min = B["unravel"](jnp.asarray(prev["theta_min"]))
        if do_cold:
            theta_max = B["unravel"](jnp.asarray(prev["theta"][prev["cycle"]==0][-1]))
            out_path = "runs/wmam_readout_cold.npz"
        if verbose:
            print(f"[readout] recalled theta_max/theta_min from {load}")
    else:
        if verbose:
            print("[readout] no --load; running a cyclic pass first for theta_max")
        clog = cyclic_train_logged(cfg, verbose=verbose, out_path=None, stride=stride)
        theta_max, theta_min = clog["_theta_max_tree"], clog["_theta_min_tree"]
    
    near = B["near"]
    span = cfg.tau_max - cfg.tau_min
    grid = np.round(np.arange(cfg.tau_min - 0.6 * span,
                              cfg.tau_max + 0.6 * span + 1e-9,
                              cfg.increment * 1.0), 6)
    inside = (grid >= cfg.tau_min - 1e-9) & (grid <= cfg.tau_max + 1e-9)
    dummy = jax.random.PRNGKey(0)
    p_a = probs_test(theta_max)

    idx_c, tau_c, leg_c, loss_c, th_rows = [], [], [], [], []

    def rec(bt, bl, ns, idx, tau, leg):
        bt, bl = _truncate(bt, bl, ns, stride)
        k = len(bl)
        idx_c.extend([idx] * k); tau_c.extend([tau] * k)
        leg_c.extend([leg] * k); loss_c.extend(bl.tolist()); th_rows.append(bt)

    fb, rev, rev_near, dtheta, dW, perp = [], [], [], [], [], []
    weight_max = reparam_alt(theta_max, B["knots"], B["scales"], B["mask"])
    for idx, tr in enumerate(grid):
        fb.append(float(budget(theta_max, float(tr))))
        th_r, ns1, _, bt1, bl1 = ro_log(theta_max, float(tr), dummy)
        rec(bt1, bl1, ns1, idx, float(tr), "to")
        th_dd, ns2, _, bt2, bl2 = ro_log(th_r, cfg.tau_max, dummy)
        rec(bt2, bl2, ns2, idx, cfg.tau_max, "back")
        dp = probs_test(th_dd) - p_a
        rev.append(_rms(jnp.asarray(dp)))
        rev_near.append(_rms(jnp.asarray(dp[near])) if near.any() else np.nan)
        dtheta.append(_rms(jnp.asarray(flatf(th_dd) - flatf(theta_max))))
        dW.append(_rms(jnp.asarray(flatf(reparam_alt(th_dd, B["knots"], B["scales"], B["mask"])) - flatf(weight_max))))
        perp.append(theta_perp(jnp.asarray(flatf(th_r)),
                               jnp.asarray(flatf(theta_max)),
                               jnp.asarray(flatf(theta_min))))

    obs = dict(tau_read=grid, inside=inside, fixed_budget_loss=np.array(fb),
               rev_prob=np.array(rev), rev_prob_near=np.array(rev_near),
               dtheta_rp=np.array(dtheta),
               dweights_rp=np.array(dW), theta_perp=np.array(perp))
    log = dict(
        kind="readout", seed=cfg.seed,
        cycle=np.array(idx_c), tau=np.array(tau_c),
        leg=np.array(leg_c, dtype="<U8"), loss=np.array(loss_c),
        theta=np.concatenate(th_rows, axis=0), kink=B["kink"],
        theta_cold=flatf(theta_max), theta_max=flatf(theta_max),
        theta_min=flatf(theta_min), grid=grid, inside=inside,
        **{f"ro_{k}": v for k, v in obs.items()})
    if out_path:
        save_log(log, out_path, meta=dict(cfg=cfg.as_dict()))
#       if verbose:
#           def con(a):
#               a = a[~np.isnan(a)] if np.isnan(a).any() else a
#               return float(a[~inside].mean() / max(a[inside].mean(), 1e-30))
#           print(f"[readout] outside/inside contrast: "
#                 f"retrain={con(obs['fixed_budget_loss']):.1f}x  "
#                 f"rev(all)={con(obs['rev_prob']):.1f}x  "
#                 f"rev(near)={con(obs['rev_prob_near']):.1f}x")
#           print(f"[readout] {log['theta'].shape[0]} steps logged -> {out_path}")

    return log, obs


# ----------------------------------------------------------------------------
# Save / recall  and  per-group plotting
# ----------------------------------------------------------------------------
def save_log(log, path, meta=None):
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    arrays = {k: v for k, v in log.items()
              if isinstance(v, np.ndarray) or np.isscalar(v)}
    arrays["_kind"] = np.array(log.get("kind", "cyclic"))
    arrays["_meta"] = np.array(json.dumps(meta or {}))
    np.savez_compressed(path, **arrays)
    return path


def load_log(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if not k.startswith("_")}
    out["kind"] = str(z["_kind"]) if "_kind" in z.files else "cyclic"
    if "_meta" in z.files:
        out["meta"] = json.loads(str(z["_meta"]))
    return out



# ----------------------------------------------------------------------------
# Entry point: one flag chooses cyclic training or readout
# ----------------------------------------------------------------------------
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Per-step-logged cyclic training OR readout for the "
                    "weight-adapting MAM.")
    p.add_argument("--bulk-optimizer", type=str, default="adam")
    p.add_argument("--further", action="store_true",
                   help="run further bulk cyclic training")
    p.add_argument("--refine", action="store_true",
                   help="run the refine instead of bulk cyclic training")
    p.add_argument("--readout", action="store_true",
                   help="run the readout instead of cyclic training")
    p.add_argument("--docold", action="store_true",
                   help="run the readout for cold cycle")
    p.add_argument("--load", default=None,
                   help="readout: recall theta_max/theta_min from a saved "
                        "cyclic log (else a cyclic pass is run first)")
    p.add_argument("--out", default=None, help="output .npz (auto-named if omitted)")
    p.add_argument("--stride", type=int, default=1,
                   help=">1 thins stored steps")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--kink-fraction", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--maxsteps", type=int, default=None)
    p.add_argument("--setup", default=None,
                   help="path to a frozen setup .npz: loaded if it exists "
                        "(guarantees identical data/init/kinks), else built "
                        "and saved there")
    return p.parse_args()


def main():
    a = _parse_args()
    overrides = {}
    if a.seed is not None:
      overrides["seed"] = a.seed
    if a.lr is not None:
      overrides["refine_lr"] = a.lr
    if a.tol is not None:
      overrides["refine_tol"] = a.tol
    if a.maxsteps is not None:
      overrides["refine_max_steps"] = a.maxsteps
    overrides["bulk_optimizer"] = a.bulk_optimizer
    overrides["kink_fraction"] = a.kink_fraction
    cfg = WMAMConfig(**overrides)
    print("kink fraction:{}".format(cfg.kink_fraction))
    print("maxstep:{}".format(cfg.refine_max_steps))
    #cfg = WMAMConfig() if a.seed is None else WMAMConfig(seed=a.seed)

    B = None
    if a.setup:
        if os.path.exists(a.setup):
            B, _ = load_setup(a.setup)        # frozen data/init/kinks + cfg
            print(f"[setup] loaded frozen setup from {a.setup}")
        else:
            B = _build(cfg)
            save_setup(B, cfg, a.setup)
            print(f"[setup] built and froze setup -> {a.setup}")

    if a.readout:
        readout_logged(cfg, load=a.load, stride=a.stride, do_cold=a.docold,
                       setup=B, grad_clip=a.grad_clip,
                       out_path=a.out or "runs/wmam_readout.npz")
    elif a.refine:
        refine_cyclic_train_logged(cfg, load=a.load, stride=a.stride, setup=B,
                       further=a.further, grad_clip=a.grad_clip,
                       out_path=a.out or "runs/wmam_refine_cyclic.npz")
    else:
        cyclic_train_logged(cfg, stride=a.stride, setup=B, grad_clip=a.grad_clip,
                            out_path=a.out or "runs/wmam_cyclic.npz")


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    main()
