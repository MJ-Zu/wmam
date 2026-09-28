"""Kinked reparameterization for "special" parameters.

The whole point of this module is to inject *controlled discontinuities in the
gradient* (not in the loss) at chosen parameters, while leaving every other
parameter smooth.

For a special parameter with raw (stored, optimized) value ``x`` we feed the
network the *effective* value

    x' = x + s * |x - c|

instead of ``x`` itself. Consequences:

    * x' is CONTINUOUS in x  (the loss stays continuous -- x' only has a corner)
    * dx'/dx = 1 + s * sign(x - c)  jumps from (1 - s) to (1 + s) across x = c,
      i.e. the gradient dL/dx has a jump discontinuity of size 2*s * (dL/dx')

Design notes (decided in the design discussion):

    * The linear ``x`` term breaks the reflection symmetry of a pure ``s|x-c|``
      V (there, x and 2c-x map to the same x'). With the linear term the two
      sides carry different slopes (1-s vs 1+s), so the map is injective for
      |s| < 1 and the jump size 2s is cleanly decoupled from everything else.

    * |s| controls the REGIME:
        - |s| < 1 : both slopes share a sign  -> x' is monotonic through c,
                    the corner is a mere kink the parameter *crosses*.
        - |s| > 1 : the slopes have opposite signs -> x' has a local extremum
                    at c, so c can be an *attractor*: the parameter is driven
                    onto the corner and chatters there ("continuously active"
                    kink). THIS is the regime we want, so we sample |s| in
                    (s_low, s_high) with s_low > 1.
      Whether c actually attracts also depends on the sign of the downstream
      gradient dL/dx' (an emergent, data-dependent property we do not control
      per-parameter); |s| > 1 is the structural precondition, not a guarantee.

    * The knot ``c`` is anchored to each parameter's *initial value* -> every
      special parameter has a kink pinned to where it was born. (Set
      ``knot_mode="zero"`` to put every corner at the origin instead.)

    * Which parameters are special (the mask) and their (c, s) are drawn ONCE
      with a fixed seed and then held constant for the entire tau-chain. If
      they resampled between tau steps, the warm-start comparison would be
      confounded (architecture changing at the same time as the label
      threshold).

Implementation note: at exactly x = c, ``jax.grad(jnp.abs)`` returns +1 (the
right derivative -- a subgradient pick, NOT 0), so dx'/dx = 1 + s there. It is a
measure-zero point and does not affect dynamics; the discontinuity shows up as
the gradient flipping sign as x crosses c between steps, not as a runtime error.

A special parameter is engineered never to have a vanishing gradient, so it
never "settles". Any convergence / plateau / early-stopping logic must watch
the loss or the smooth parameters -- never these. See train.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def reparam(params, knots, scales):
    """Map raw params -> effective params: x' = x + s * |x - c|, elementwise.

    ``knots`` (c) and ``scales`` (s) are pytrees with the *same structure* as
    ``params``. For non-special elements ``s == 0``, so x' = x exactly.
    """
    return jax.tree_util.tree_map(
        lambda x, c, s: x + s * jnp.abs(x - c),
        params, knots, scales,
    )

def reparam_alt(params, knots, scales, mask):
    """Kink map with the linear term dropped for special params only.

    special (mask=True):  x' = s * |x - c|     (drops the linear x term)
    normal  (mask=False): x' = x               (identity, unchanged)
    """
    def f(x, c, s, m):
        kinked = s * jnp.abs(x - c)
        return jnp.where(m, kinked, x)

    return jax.tree_util.tree_map(f, params, knots, scales, mask)


def build_kink_params(params, key, fraction, s_low=1.5, s_high=3.0,
                      knot_mode="init"):
    """Draw the fixed kink configuration once, for the whole chain.

    Args:
        params: an initialized parameter pytree (defines structure + init vals).
        key: a jax PRNGKey. Fix it across the chain for reproducibility.
        fraction: probability that any given scalar parameter is "special".
        s_low, s_high: |s| is drawn uniformly from this range. Keep s_low > 1
            to stay in the attractor ("continuously active") regime.
        knot_mode: "init"  -> c = the parameter's initial value (default),
                   "zero"  -> c = 0 for every parameter.

    Returns:
        (knots, scales, mask) pytrees, each matching ``params`` structure.
        ``mask`` is boolean (True = special) and is only for bookkeeping/reports.
    """
    #assert s_low > 1.0, "s_low must exceed 1 to keep corners as attractors"

    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))

    scale_leaves, mask_leaves, knot_leaves = [], [], []
    for leaf, k in zip(leaves, keys):
        k_mask, k_mag, k_sign = jax.random.split(k, 3)
        m = jax.random.bernoulli(k_mask, p=fraction, shape=leaf.shape)
        mag = jax.random.uniform(k_mag, leaf.shape, minval=s_low, maxval=s_high)
        sign = jnp.where(jax.random.bernoulli(k_sign, 0.5, leaf.shape), 1.0, -1.0)
        s = jnp.where(m, mag * sign, 0.0).astype(leaf.dtype)

        c = jnp.zeros_like(leaf) if knot_mode == "zero" else leaf

        scale_leaves.append(s)
        mask_leaves.append(m)
        knot_leaves.append(c)

    scales = jax.tree_util.tree_unflatten(treedef, scale_leaves)
    mask = jax.tree_util.tree_unflatten(treedef, mask_leaves)
    knots = jax.tree_util.tree_unflatten(treedef, knot_leaves)
    return knots, scales, mask


def kink_stats(mask):
    """Return (#special, #total, fraction) for logging."""
    leaves = jax.tree_util.tree_leaves(mask)
    n_special = int(sum(int(jnp.sum(m)) for m in leaves))
    n_total = int(sum(int(m.size) for m in leaves))
    frac = n_special / max(n_total, 1)
    return n_special, n_total, frac


def pick_watched_coords(scales, knots, n_watch=6, seed=0):
    """Choose n_watch special scalar coordinates to track over training.

    Returns a list of dicts with the leaf index, the coordinate tuple, and that
    coordinate's knot c and scale s. Shared by the chain's per-step recorder and
    the standalone investigator so both watch coordinates the same way.
    """
    import numpy as np
    flat_s = jax.tree_util.tree_leaves(scales)
    flat_c = jax.tree_util.tree_leaves(knots)
    rng = np.random.default_rng(seed)
    pool = []
    for li, s in enumerate(flat_s):
        for coord in np.argwhere(np.asarray(s) != 0.0):
            pool.append((li, tuple(int(v) for v in coord)))
    rng.shuffle(pool)
    watched = []
    for li, coord in pool[:n_watch]:
        watched.append({"leaf": li, "coord": coord,
                        "c": float(np.asarray(flat_c[li])[coord]),
                        "s": float(np.asarray(flat_s[li])[coord])})
    return watched


def read_coords(params, watched):
    """Read the current values of the watched coordinates as a numpy array."""
    import numpy as np
    flat = jax.tree_util.tree_leaves(params)
    return np.array([float(np.asarray(flat[w["leaf"]])[w["coord"]])
                     for w in watched])
