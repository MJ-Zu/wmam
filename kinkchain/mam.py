"""Minimal near-critical harness to test whether a *learned* system forms a
Marginally Absorbing Manifold (MAM) via Gradient Discontinuity Learning (GDL),
following Zu & Goodrich.

This deliberately mirrors the physical packing rather than the ResNet:

  * Small parameter vector theta in R^n (n ~ tens), analogous to the species
    diameters -- near-critically constrained, so there is little null space to
    drift in.
  * A single scalar observable F(theta), analogous to the Poisson ratio,
    computed by a FIXED random smooth function (a tiny tanh net). theta is the
    only trainable thing (inverse design).
  * Engineered gradient discontinuities via the kink reparam theta' = theta +
    s|theta - c|, playing the role of contact changes. tanh is smooth, so the
    kinks are the ONLY GDs.
  * DETERMINISTIC full-gradient descent (no mini-batches, no noise), because
    noise gives transient memories, not the clean MAM we want to see first.
  * Cyclic training of the target tau between tau_min and tau_max.

Readout (the analog of Fig. 2): from the converged state at tau_max, train to a
readout target tau_read and back, and measure return-point quantities vs
tau_read. A MAM shows: reversibility (Delta-theta_RP ~ 0) and easy training
(few steps) WITHIN [tau_min, tau_max], with sharp jumps/kinks just outside.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax.tree_util import Partial

jax.config.update("jax_enable_x64", True)   # tight return-point measurements


@dataclass
class MAMConfig:
    n: int = 32                 # trainable parameters (species-diameter analog)
    m: int = 16                 # hidden units of the fixed random F net
    kink_fraction: float = 0.4  # fraction of theta with an engineered kink
    s_low: float = 1.5          # |s| ~ U(s_low, s_high); >1 => attractor regime
    s_high: float = 3.0
    v_scale: float = 0.5        # sets the achievable range of F
    tau_min: float = -0.5
    tau_max: float = 0.5
    increment: float = 0.1      # target step per training within a cycle
   # --- bulk phase ---
    # NB: for this smooth scalar-F harness, plain GD forms the cleanest MAM.
    # Adaptive optimizers (rmsprop/adam) inject perpendicular drift that keeps
    # dD erratic and prevents a clean absorbing manifold (paper Sec IV C). They
    # are selectable for comparison, but "gd" is the default that works here.
    bulk_optimizer: str = "rmsprop"        # "gd" | "rmsprop" | "adam"
    lr: float = 1.e-4                  # bulk step size (large enough for plain GD)
    bulk_afactor: float = 0.96        # bulk afactor is a decay factor for bulk step size
    n_cycles: int = 50                # max bulk cycles (stops early on MAM-approach)
    max_steps: int = 500
    tol: float = 1e-24                # per-target convergence tol (bulk)
    # --- MAM-approach detection (ends the bulk phase) ---
    converge_tol: float = 1e-24        # a cycle "converges" if EVERY target's final loss < this
    dD_patience: int = 3              # number of recent cycles that must have stable dD
    dD_plateau_rel: float = 1.e-1      # dD is "plateaued" if its relative spread < this
    # --- refine phase + readout (plain fixed small-lr GD: clean measurement) ---
    refine_cycles: int = 10
    refine_lr: float = 2e-5
    refine_tol: float = 1e-20
    refine_max_steps: int = 100000
    seed: int = 1110

    def as_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------
# The fixed problem: random smooth F + fixed kink config
# ----------------------------------------------------------------------------
def build_problem(cfg: MAMConfig):
    rng = np.random.default_rng(cfg.seed)
    theta0 = rng.standard_normal(cfg.n)
    # kink config, FIXED for the whole experiment; knots anchored at init
    kmask = rng.random(cfg.n) < cfg.kink_fraction
    mag = rng.uniform(cfg.s_low, cfg.s_high, cfg.n)
    sign = np.where(rng.random(cfg.n) < 0.5, 1.0, -1.0)
    s = np.where(kmask, mag * sign, 0.0)
    c = theta0.copy()
    # fixed random smooth readout net F(theta') = v . tanh(W theta' + b)
    W = rng.standard_normal((cfg.m, cfg.n)) / np.sqrt(cfg.n)
    b = 0.1 * rng.standard_normal(cfg.m)
    v = cfg.v_scale * rng.standard_normal(cfg.m)
    P = {k: jnp.asarray(x) for k, x in
         dict(theta0=theta0, s=s, c=c, W=W, b=b, v=v).items()}
    P["n_special"] = int(kmask.sum())
    return P


def F(theta, P):
    """Scalar observable. Kink reparam then a fixed smooth tanh net."""
    tp = theta + P["s"] * jnp.abs(theta - P["c"])
    return jnp.dot(P["v"], jnp.tanh(P["W"] @ tp + P["b"]))


# ----------------------------------------------------------------------------
# Deterministic training to a scalar target (jitted while-loop)
# ----------------------------------------------------------------------------
def make_trainer(P, max_steps, tol, opt_name="gd", lr=1.e-2, afactor=0.1):
    """Train theta to a scalar target by descent. opt_name selects the inner
    optimizer: 'gd' (plain fixed-lr SGD -- steepest descent, clean for the
    refine phase and readout), or 'rmsprop'/'adam' (adaptive -- robust and fast,
    for the bulk phase; a single un-tuned lr converges across the uneven,
    kinked landscape). Returns (params, n_steps, final_loss)."""
    def loss_fn(theta, tau):
        return (F(theta, P) - tau) ** 2
    grad_fn = jax.grad(loss_fn)

    learning_rate_scheduler = optax.cosine_decay_schedule(lr, max_steps, afactor)
    if opt_name != "gd":
      opt = {"adam": optax.adam,
            "rmsprop": optax.rmsprop}[opt_name](learning_rate_scheduler)

    @jax.jit
    def gd_train(theta, tau):
        def cond(carry):
            _, i, l = carry
            return (i < max_steps) & (l > tol)
        def body(carry):
            th, i, _ = carry
            th = th - lr * grad_fn(th, tau)
            return th, i + 1, loss_fn(th, tau)
        init = (theta, 0, loss_fn(theta, tau))
        th, i, l = jax.lax.while_loop(cond, body, init)
        max_grad = jnp.amax(jnp.abs(grad_fn(th, tau)))
        return th, i, l, lr, max_grad

    @jax.jit
    def train(theta, tau):
        st0 = opt.init(theta)
        def cond(carry):
            _, _, i, l = carry
            return (i < max_steps) & (l > tol)
        def body(carry):
            th, s, i, _ = carry
            updates, s = opt.update(grad_fn(th, tau), s, th)
            th = optax.apply_updates(th, updates)
            return th, s, i + 1, loss_fn(th, tau)
        th, s, i, l = jax.lax.while_loop(
            cond, body, (theta, st0, 0, loss_fn(theta, tau)))
        lr = learning_rate_scheduler(i)
        max_grad = jnp.amax(jnp.abs(grad_fn(th, tau)))
        return th, i, l, lr, max_grad

    if opt_name == "gd":
    	return gd_train
    else:
        return train


def _rms(x):
    return float(jnp.sqrt(jnp.mean(x ** 2)))


def theta_perp(theta, th_max, th_min):
    """Perpendicular distance from theta to the line through th_max, th_min
    (Eq. 2 of the paper)."""
    vartheta = th_min - th_max
    d = theta - th_max
    denom = jnp.dot(vartheta, vartheta) + 1e-30
    proj = (jnp.dot(d, vartheta) / denom) * vartheta
    return float(jnp.linalg.norm(d - proj))


# ----------------------------------------------------------------------------
# Cyclic training
# ----------------------------------------------------------------------------
def _triangle_targets(cfg):
    """One cycle of targets: tau_max -> tau_min -> tau_max (excluding the
    starting tau_max, which is where each cycle begins)."""
    tmax, tmin, inc = cfg.tau_max, cfg.tau_min, cfg.increment
    down = list(np.round(np.arange(tmax - inc, tmin - 1e-9, -inc), 6))
    up = list(np.round(np.arange(tmin + inc, tmax + 1e-9, inc), 6))
    return down, up

def _run_cycle(train, theta, down, up, traj, traj_cycle, cyc):
    """One triangle cycle; returns theta, theta_at_tau_min, worst final loss,
    worst step count (the last two are the per-cycle convergence diagnostics)."""
    max_loss, max_ns = 0.0, 0
    for t in down:
        theta, ns, l, lr, maxgrad = train(theta, float(t))
        traj.append(np.asarray(theta)); traj_cycle.append(cyc)
        max_loss = max(max_loss, float(l)); max_ns = max(max_ns, int(ns))
    theta_min = theta
    for t in up:
        theta, ns, l, lr, maxgrad = train(theta, float(t))
        traj.append(np.asarray(theta)); traj_cycle.append(cyc)
        max_loss = max(max_loss, float(l)); max_ns = max(max_ns, int(ns))
    return theta, theta_min, max_loss, max_ns, lr, maxgrad
 
 
def _dD_plateaued(dD, patience, rel):
    """True if the last `patience` per-cycle dD values are stable (small
    relative spread) -- i.e. dD has stopped changing."""
    if len(dD) < patience:
        return False
    recent = np.array(dD[-patience:])
    return (recent.max() - recent.min()) / (recent.mean() + 1e-30) < rel
 
 
def cyclic_train(cfg, P, verbose=False):
    """Two phases:
      Phase 1 (bulk): adaptive optimizer, run cycles until the MAM is approached
        -- detected when BOTH (1) every target in the cycle converged
        (max final loss < converge_tol) AND (2) dD has plateaued -- or until the
        n_cycles cap.
      Phase 2 (refine): plain fixed small-lr GD for cfg.refine_cycles, so the
        final state and the readout are steepest-descent-clean and any
        irreversibility is attributable to the engineered kinks, not the
        optimizer.
    """
    down, up = _triangle_targets(cfg)
    bulk = make_trainer(P, cfg.max_steps, cfg.tol,
                        opt_name=cfg.bulk_optimizer, 
                        lr=cfg.lr,
                        afactor=cfg.bulk_afactor)
 
    theta = P["theta0"]
    theta, _, _, _, _ = bulk(theta, cfg.tau_max)       # initial training to tau_max
    theta_cold = theta                            # cold (before-cycling) anchor
 
    traj, traj_cycle = [np.asarray(theta)], [0]
    dD, maxloss, maxsteps, prev = [], [], [], theta
    theta_min = None
    mam_cycle = -1
    lr = cfg.lr
    count_converge = 0
 
    # ---- Phase 1: bulk cycles until the MAM is approached ----
    for cyc in range(1, cfg.n_cycles + 1):
        afactor = jnp.where(lr<=cfg.refine_lr*0.1, 1.0, cfg.bulk_afactor)
        lr = max(lr, cfg.refine_lr*0.1)
        bulk = make_trainer(P, cfg.max_steps, cfg.tol,
                        opt_name=cfg.bulk_optimizer, 
                        lr=lr,
                        afactor=afactor)
        theta, theta_min, ml, ms, lr, maxgrad = _run_cycle(bulk, theta, down, up,
                                               traj, traj_cycle, cyc)
        dD.append(_rms(theta - prev)); prev = theta
        maxloss.append(ml); maxsteps.append(ms)
        converged = ml < cfg.converge_tol
        plateaued = _dD_plateaued(dD, cfg.dD_patience, cfg.dD_plateau_rel)
        if verbose and (cyc == 1 or cyc % 5 == 0):
            print(f"  bulk {cyc:3d}  dD={dD[-1]:.2e}  max_loss={ml:.1e}  lr={lr:.3e}  max_grad={maxgrad:.3e}  "
                  f"max_steps={ms}  converged={converged}  dD_plateau={plateaued}")
        if converged and plateaued:
            mam_cycle = cyc
            count_converge += 1
            if verbose:
                print(f"  -> MAM approached at bulk cycle {cyc} "
                      f"(all targets converged, dD plateaued)")
            if count_converge > cfg.dD_patience :
##and lr<cfg.refine_lr:
                break
        else:
            count_converge = 0
    if mam_cycle < 0 and verbose:
        print(f"  (warning: MAM-approach not detected within {cfg.n_cycles} "
              f"bulk cycles; proceeding to refine anyway)")
    n_bulk = len(dD)
 
    # ---- Phase 2: refine with plain fixed small-lr GD ----
    refine = make_trainer(P, cfg.refine_max_steps,
                          cfg.refine_tol, opt_name="gd", lr=cfg.refine_lr)
    for r in range(cfg.refine_cycles):
        theta, theta_min, ml, ms, lr, maxgrad = _run_cycle(refine, theta, down, up,
                                               traj, traj_cycle, n_bulk + 1 + r)
        dD.append(_rms(theta - prev)); prev = theta
        maxloss.append(ml); maxsteps.append(ms)
        if verbose:
            print(f"  refine {r + 1:2d}  dD={dD[-1]:.2e}  max_loss={ml:.1e}  lr={lr:.3e}, max_grad={maxgrad:.3e} "
                  f"max_steps={ms}")
 
    return dict(
        theta_cold=theta_cold, theta_max=theta, theta_min=theta_min,
        traj=np.stack(traj), traj_cycle=np.array(traj_cycle),
        dD=np.array(dD), maxloss=np.array(maxloss), maxsteps=np.array(maxsteps),
        n_bulk=n_bulk, mam_cycle=mam_cycle)
 
 
# ----------------------------------------------------------------------------
# Return-point readout (analog of Fig. 2)
# ----------------------------------------------------------------------------
def readout(cfg, P, theta_start, th_max_ref, th_min_ref, tau_read_grid):
    """From theta_start (at tau_max), for each tau_read: train to tau_read
    (record n_steps and theta_perp), then train back to tau_max (record the
    return-point change Delta-theta_RP). Uses the SAME plain fixed small-lr GD
    as the refine phase, so the measurement is steepest-descent-clean."""
    train = make_trainer(P, cfg.refine_max_steps,
                         cfg.refine_tol, opt_name="gd", lr=cfg.refine_lr)
    nsteps, dRP, perp, Fread = [], [], [], []
    for tr in tau_read_grid:
        th_r, n1, _, _, _= train(theta_start, float(tr))
        th_back, _, _, _, _ = train(th_r, cfg.tau_max)
        nsteps.append(int(n1))
        dRP.append(_rms(th_back - theta_start))
        perp.append(theta_perp(th_r, th_max_ref, th_min_ref))
        Fread.append(float(F(th_r, P)))
    return dict(tau_read=np.asarray(tau_read_grid), nsteps=np.array(nsteps),
                dtheta_rp=np.array(dRP), theta_perp=np.array(perp),
                F_at_read=np.array(Fread))
 
 
def run_mam(cfg: MAMConfig | None = None, verbose=True, out_path=None, do_cold=True):
    cfg = cfg or MAMConfig()
    P = build_problem(cfg)
    if verbose:
        print(f"[mam] n={cfg.n} params, {P['n_special']} kinked; "
              f"F range target [{cfg.tau_min}, {cfg.tau_max}]")
    res = cyclic_train(cfg, P, verbose=verbose)
 
    # readout grid spans and exceeds the training range
    span = cfg.tau_max - cfg.tau_min
    grid = np.round(np.arange(cfg.tau_min - 0.6 * span,
                              cfg.tau_max + 0.6 * span + 1e-9, cfg.increment), 6)
    ref_max, ref_min = res["theta_max"], res["theta_min"]
    ro_mam = readout(cfg, P, res["theta_max"], ref_max, ref_min, grid)
    ro_cold = (readout(cfg, P, res["theta_cold"], ref_max, ref_min, grid)
               if do_cold else ro_mam)
 
    inside = (grid >= cfg.tau_min - 1e-9) & (grid <= cfg.tau_max + 1e-9)
    outside = ~inside
    if verbose:
        din = ro_mam["dtheta_rp"][inside].mean()
        dout = ro_mam["dtheta_rp"][outside].mean()
        print(f"[mam] converged: dD {res['dD'][0]:.2e} -> {res['dD'][-1]:.2e}")
        print(f"[mam] MAM return-point Delta-theta_RP:  inside={din:.2e}  "
              f"outside={dout:.2e}  ratio={dout / max(din, 1e-30):.1f}x")
    out = dict(cfg=cfg.as_dict(), problem_n_special=P["n_special"],
                **res, readout_cold=ro_cold, readout_mam=ro_mam,
                inside=inside, grid=grid)
    if out_path:
      np.savez(out_path, **{k: v for k, v in out.items() if k!="cfg"})
      if verbose:
        print("f[mam] saved {out_path}")
    return out
 
 
def memory_metrics(res):
    """Summarize memory strength from a run: inside/outside contrast of the two
    strong return-point signals. Higher ratio = stronger memory."""
    ins = res["inside"]; out = ~ins
    m = res["readout_mam"]
    def contrast(a):
        vi = float(np.mean(a[ins])); vo = float(np.mean(a[out]))
        return vo / max(vi, 1e-30), vi, vo
    r_d, di, do = contrast(m["dtheta_rp"])
    r_p, pi, po = contrast(m["theta_perp"])
    return dict(ratio_dtheta=r_d, inside_dtheta=di, outside_dtheta=do,
                ratio_perp=r_p, inside_perp=pi, outside_perp=po)
 
 
if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    run_mam(out_path="runs/mam.npz")
