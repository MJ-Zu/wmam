"""Setup files: one JSON or pickle file describes a whole training run.

A training run needs
  1. a configuration (the WMAMConfig fields),
  2. the frozen setup -- the dataset, the initial weights, the kink knots /
     amplitudes / mask,
  3. a few run options (which phase to run, tau, gradient clip, logging stride, ...).
They all live in ONE file, so a training command only needs the input (setup)
path and the output path:

    python -m kinkchain.run_setup --setup setups/exp1.setup.pkl --out runs/exp1_bulk.npz

Two kinds of setup file
  * pickle (*.pkl) = FROZEN. cfg + run options + the arrays, written by
    `python -m kinkchain.make_setup`. Training reads the dataset, init and kinks
    from it, so every machine (laptop, cluster, Colab) starts from exactly the
    same arrays. Arrays are stored as (dtype, shape, raw bytes) tuples, so the
    file is plain Python objects and does not depend on the numpy version.
    Only load pickles you made yourself: unpickling runs code.
  * JSON (*.json) = a recipe, human readable and editable. Layout:
        {"base": "setups/exp1.setup.pkl",    # optional: take the frozen arrays + base cfg from this pickle
         "cfg":  {...},                      # WMAMConfig fields (partial is fine; defaults fill the rest)
         "run":  {...}}                      # run options, see RUN_DEFAULTS
    Without "base" the arrays are rebuilt from the cfg (data file + seeds) on every
    run. With "base" the JSON is a stage description layered on the frozen setup:
    its cfg may change training hyper-parameters (learning rates, step budgets,
    ...) but not the fields that define the arrays (FROZEN_FIELDS).
    A JSON that is just a flat dict of WMAMConfig fields (like
    configs/base_cfg.json) is read as {"cfg": that dict, "run": {"mode": "single_tau"}}.

run options
  mode        "single_tau" (bulk -> refine at one tau), "cyclic" (cold + bulk cycles),
              "refine" (refine cycles from a previous cyclic output), "further"
              (further bulk cycles from a previous output)
  tau         single_tau only; default cfg.tau_max
  grad_clip   clip every gradient element to [-c, c] (None = off)
  stride      cyclic modes: record theta/loss every STRIDE-th step during training
  batch_seed  cyclic modes: seed offset of the mini-batch noise
  rescale     refine / further: rescale the kinked-parameter gradients (plain GD)
  load        refine / further: the previous output (.npz) to continue from
  out         default output path (the command-line --out wins)

Two configuration lineages exist and are kept as they were: "single_tau" uses
mam_weights._build / mam_weights_single_tau_train.WMAMConfig (separate seed_kink,
knots at the initial values); the cyclic modes use mam_weights_reparam_alt._build /
WMAMConfig (kink key = seed + 99, knots at 0).
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, fields

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

VERSION = 1
CYCLIC_MODES = ("cyclic", "refine", "further")
MODES = ("single_tau",) + CYCLIC_MODES
RUN_DEFAULTS = dict(mode="cyclic", tau=None, grad_clip=None, stride=1, batch_seed=12345,
                    rescale=False, load=None, out=None)

# cfg fields that define the frozen arrays; a stage JSON with a "base" may not change them
FROZEN_FIELDS = ("img", "n_train", "n_test", "channels", "kink_fraction", "s_low", "s_high",
                 "property_noise", "data", "seed", "seed_kink", "tau_max", "near_band")

# config files written before WMAMConfig was renamed / trimmed
_LEGACY_KEYS = {"lr": "bulk_lr", "max_steps": "bulk_max_steps", "tol": "bulk_tol"}
_DROPPED_KEYS = ("readout_grad_clip",)

_ARRAYS = ("Xtr", "atr", "Xte", "ate", "near", "kink")
_TREES = ("params0", "knots", "scales", "mask")


def kind_of(mode):
    if mode not in MODES:
        raise ValueError(f"run.mode must be one of {MODES}, got {mode!r}")
    return "single_tau" if mode == "single_tau" else "cyclic"


def cfg_class(mode):
    if kind_of(mode) == "single_tau":
        from .mam_weights_single_tau_train import WMAMConfig
    else:
        from .mam_weights_reparam_alt import WMAMConfig
    return WMAMConfig


def _translate(partial):
    out = {}
    for k, v in partial.items():
        if k in _DROPPED_KEYS:
            continue
        out[_LEGACY_KEYS.get(k, k)] = v
    for old, new in _LEGACY_KEYS.items():          # an explicit new-style key wins over a legacy one
        if old in partial and new in partial:
            out[new] = partial[new]
    return out


def resolve_cfg(mode, partial=None, base=None):
    """The complete cfg dict for `mode`: the class defaults (or `base`) with `partial` on top."""
    cls = cfg_class(mode)
    d = dict(base) if base is not None else asdict(cls())
    d.update(_translate(partial or {}))
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(d) - set(names))
    if unknown:
        raise ValueError(f"unknown config field(s) for mode {mode!r}: {unknown}")
    cls(**d)                                        # validates the field set
    return {k: d[k] for k in names}


def build_B(mode, cfg_dict):
    """The arrays-and-model bundle, built from the cfg (data file + seeds)."""
    cfg = cfg_class(mode)(**cfg_dict)
    if kind_of(mode) == "single_tau":
        from .mam_weights import _build
    else:
        from .mam_weights_reparam_alt import _build
    return _build(cfg)


def _pack(a):
    a = np.ascontiguousarray(np.asarray(a))
    return (a.dtype.str, tuple(a.shape), a.tobytes())


def _unpack(t):
    dt, shape, buf = t
    return np.frombuffer(buf, dtype=np.dtype(dt)).reshape(shape).copy()


def save_setup(B, cfg_dict, run, path, json_copy=True):
    """Freeze B + cfg + run options into a pickle (and, by default, a readable
    <stem>.cfg.json copy of cfg + run next to it)."""
    run = {**RUN_DEFAULTS, **run}
    kind_of(run["mode"])
    arrays = {k: _pack(B[k]) for k in _ARRAYS}
    arrays.update({k: _pack(B["flat"](B[k])) for k in _TREES})
    payload = {"version": VERSION, "kind": kind_of(run["mode"]), "cfg": cfg_dict, "run": run, "arrays": arrays}
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    if json_copy:
        stem = path[:-4] if path.endswith(".pkl") else path
        with open(stem + ".cfg.json", "w") as f:
            json.dump({"cfg": cfg_dict, "run": run}, f, indent=2, sort_keys=True)
    return path


def _rebuild_B(cfg_dict, arrays):
    from .mam_weights_reparam_alt import TinyCNN
    Xtr = arrays["Xtr"]
    model = TinyCNN(channels=cfg_dict["channels"])
    template = model.init(jax.random.PRNGKey(cfg_dict["seed"]), jnp.asarray(Xtr[:1]))["params"]
    _, unravel = ravel_pytree(template)                 # structure only; the values are unused
    B = dict(Xtr=jnp.asarray(Xtr), atr=jnp.asarray(arrays["atr"]),
             Xte=jnp.asarray(arrays["Xte"]), ate=jnp.asarray(arrays["ate"]),
             near=arrays["near"], kink=arrays["kink"], model=model, unravel=unravel,
             flat=lambda p: np.asarray(ravel_pytree(p)[0]))
    for k in _TREES:
        B[k] = unravel(jnp.asarray(arrays[k]))
    return B


def _load_pickle(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("version") != VERSION:
        raise ValueError(f"{path}: setup version {payload.get('version')!r}, this code reads {VERSION}")
    arrays = {k: _unpack(v) for k, v in payload["arrays"].items()}
    return payload, arrays


def _read_json(path):
    with open(path) as f:
        d = json.load(f)
    if not ({"cfg", "run", "base"} & set(d)):                 # flat cfg file
        d = {"cfg": d, "run": {"mode": "single_tau"}}
    return d


def load_setup(path):
    """Read a setup file (.pkl frozen or .json recipe). Returns (B, cfg_dict, run)."""
    if path.endswith(".json"):
        d = _read_json(path)
        run = {**RUN_DEFAULTS, **d.get("run", {})}
        if "base" in d:
            payload, arrays = _load_pickle(d["base"])
            run = {**RUN_DEFAULTS, **payload["run"], **d.get("run", {})}
            mode = run["mode"]
            if kind_of(mode) != payload["kind"]:
                raise ValueError(f"{path}: mode {mode!r} needs a {kind_of(mode)!r} setup but {d['base']} is {payload['kind']!r}")
            over = _translate(d.get("cfg", {}))
            bad = sorted(k for k in over if k in FROZEN_FIELDS and k in payload["cfg"] and over[k] != payload["cfg"][k])
            if bad:
                raise ValueError(f"{path}: cfg field(s) {bad} define the frozen arrays in {d['base']}; "
                                 f"build a new setup (make_setup) instead of overriding them")
            cfg_dict = resolve_cfg(mode, over, base=payload["cfg"])
            return _rebuild_B(payload["cfg"], arrays), cfg_dict, run
        mode = run["mode"]
        cfg_dict = resolve_cfg(mode, d.get("cfg", {}))
        return build_B(mode, cfg_dict), cfg_dict, run
    payload, arrays = _load_pickle(path)
    return _rebuild_B(payload["cfg"], arrays), payload["cfg"], {**RUN_DEFAULTS, **payload["run"]}


def describe(path):
    """Printable summary of a setup file without building anything."""
    if path.endswith(".json"):
        return json.dumps(_read_json(path), indent=2, sort_keys=True)
    payload, arrays = _load_pickle(path)
    lines = [f"{path}: version {payload['version']}, kind {payload['kind']}",
             "run: " + json.dumps(payload["run"], sort_keys=True),
             "cfg: " + json.dumps(payload["cfg"], sort_keys=True),
             "arrays: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in arrays.items())]
    return "\n".join(lines)


def npz_to_setup(npz_path, out_path, run=None):
    """Convert an old .npz setup (from the previous save_setup) into a pickle.
    Those files were written by the cyclic scripts, so mode defaults to "cyclic"."""
    z = np.load(npz_path, allow_pickle=False)
    run = {**RUN_DEFAULTS, **(run or {})}
    if kind_of(run["mode"]) != "cyclic":
        raise ValueError("only cyclic-lineage .npz setups can be converted")
    cfg_dict = resolve_cfg(run["mode"], json.loads(str(z["_cfg"])))
    arrays = {k: _pack(z[k]) for k in _ARRAYS + _TREES}
    payload = {"version": VERSION, "kind": "cyclic", "cfg": cfg_dict, "run": run, "arrays": arrays}
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path
