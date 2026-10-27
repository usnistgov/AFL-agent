"""Save and load torch models by reference string.

Two functions, :func:`save_model` and :func:`load_model`, move a model between
pipeline operations without the operations ever holding a non-JSON object in
their constructor arguments. An extrapolator saves its trained model and writes
the returned *ref* string into the dataset; a downstream op reads the ref from
the dataset and loads the model.

A saved model is a :mod:`~AFL.double_agent.spec` plus a ``state_dict``. The
spec records the class and the constructor arguments needed to rebuild an
*untrained* instance (for a GP this is typically the training data). Loading
rebuilds the instance from the spec and then calls ``load_state_dict``.

Supported URI schemes
---------------------
``file:///abs/dir`` or a plain path
    One ``torch.save`` file per model in the directory.
``tiled+http://host:port/path/to/container`` (``tiled://`` is an alias)
    A child container per model holding one array per state_dict entry and per
    array-valued constructor argument, with the rest of the spec in metadata.
    Authentication uses the ``TILED_API_KEY`` environment variable if set.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import numpy as np
import torch

from AFL.double_agent.spec import SPEC_ARGS_KEY, SPEC_CLASS_KEY, class_path, resolve_spec

__all__ = ["save_model", "load_model", "is_model_ref", "clear_cache"]

_CACHE: Dict[str, torch.nn.Module] = {}

_TILED_SCHEMES = ("tiled", "tiled+http", "tiled+https")
_ARG_ARRAY_PREFIX = "arg__"
_STATE_PREFIX = "state__"


def is_model_ref(value: Any) -> bool:
    """True if ``value`` looks like a ref produced by :func:`save_model`."""
    return isinstance(value, str) and "://" in value


def clear_cache() -> None:
    """Drop all in-process cached models."""
    _CACHE.clear()


def save_model(
    model: torch.nn.Module,
    uri: str,
    args: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    builder: Optional[Callable[..., torch.nn.Module]] = None,
) -> str:
    """Save ``model`` under ``uri`` and return a unique ref string.

    Parameters
    ----------
    model
        Any object with ``state_dict``/``load_state_dict``.
    uri
        Where to store the model. See module docstring for schemes.
    args
        Constructor arguments needed to rebuild an untrained ``type(model)``.
        Values may be JSON primitives, numpy arrays, or tensors.
    name
        Optional human readable prefix for the stored key.
    builder
        Importable class or function called as ``builder(**args)`` to rebuild
        the untrained model. Defaults to ``type(model)``. Use a function when
        the model class needs non-JSON constructor arguments (e.g. BoTorch's
        ``SingleTaskGP`` with an outcome transform).

    Returns
    -------
    str
        A ref that :func:`load_model` understands. Every call produces a new ref,
        so a downstream op can never silently pick up a stale model.
    """
    args = _to_tensors(dict(args or {}))
    spec = {SPEC_CLASS_KEY: class_path(builder or type(model)), SPEC_ARGS_KEY: args}
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    key = f"{name or type(model).__name__}_{uuid.uuid4().hex[:12]}"

    scheme = urlparse(uri).scheme
    if scheme in _TILED_SCHEMES:
        ref = _save_tiled(uri, key, spec, state_dict)
    elif scheme in ("", "file"):
        ref = _save_file(uri, key, spec, state_dict)
    else:
        raise ValueError(f"Unsupported model URI scheme {scheme!r} in {uri!r}")

    _CACHE[ref] = model
    return ref


def load_model(ref: str) -> torch.nn.Module:
    """Load the model saved under ``ref`` (cached in-process by ref)."""
    if ref in _CACHE:
        return _CACHE[ref]

    scheme = urlparse(ref).scheme
    if scheme in _TILED_SCHEMES:
        spec, state_dict = _load_tiled(ref)
    elif scheme in ("", "file"):
        spec, state_dict = _load_file(ref)
    else:
        raise ValueError(f"Unsupported model ref scheme {scheme!r} in {ref!r}")

    model = resolve_spec(spec)
    model.load_state_dict(state_dict)
    model.eval()
    _CACHE[ref] = model
    return model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_tensors(args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays to tensors so the payload is loadable with weights_only=True."""
    out = {}
    for k, v in args.items():
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(np.ascontiguousarray(v))
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu()
        out[k] = v
    return out


def _file_path(uri: str) -> str:
    parsed = urlparse(uri)
    path = parsed.path if parsed.scheme == "file" else uri
    return os.path.abspath(os.path.expanduser(path))


def _save_file(uri: str, key: str, spec: dict, state_dict: dict) -> str:
    directory = _file_path(uri)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{key}.pt")
    torch.save({"spec": spec, "state_dict": state_dict}, path)
    return f"file://{path}"


def _load_file(ref: str):
    payload = torch.load(_file_path(ref), map_location="cpu", weights_only=True)
    return payload["spec"], payload["state_dict"]


def _tiled_connect(uri: str):
    from tiled.client import from_uri  # imported lazily; tiled is optional

    parsed = urlparse(uri)
    http_scheme = "https" if parsed.scheme.endswith("https") else "http"
    client = from_uri(
        f"{http_scheme}://{parsed.netloc}",
        api_key=os.environ.get("TILED_API_KEY"),
    )
    parts = [p for p in parsed.path.split("/") if p]
    return client, parts, f"{parsed.scheme}://{parsed.netloc}"


def _save_tiled(uri: str, key: str, spec: dict, state_dict: dict) -> str:
    client, parts, base = _tiled_connect(uri)
    node = client
    for part in parts:
        node = node[part] if part in node else node.create_container(key=part)

    args = spec[SPEC_ARGS_KEY]
    scalar_args = {k: v for k, v in args.items() if not isinstance(v, torch.Tensor)}
    tensor_args = {k: v for k, v in args.items() if isinstance(v, torch.Tensor)}

    model_node = node.create_container(
        key=key,
        metadata={
            "spec": {SPEC_CLASS_KEY: spec[SPEC_CLASS_KEY], SPEC_ARGS_KEY: scalar_args},
            "torch_version": torch.__version__,
        },
    )
    for name, tensor in state_dict.items():
        model_node.write_array(tensor.numpy(), key=_STATE_PREFIX + name)
    for name, tensor in tensor_args.items():
        model_node.write_array(tensor.numpy(), key=_ARG_ARRAY_PREFIX + name)

    return "/".join([base, *parts, key])


def _load_tiled(ref: str):
    client, parts, _ = _tiled_connect(ref)
    node = client
    for part in parts:
        node = node[part]

    spec = dict(node.metadata["spec"])
    args = dict(spec.get(SPEC_ARGS_KEY) or {})
    state_dict = {}
    for child_key in node:
        if child_key.startswith(_STATE_PREFIX):
            state_dict[child_key[len(_STATE_PREFIX):]] = torch.from_numpy(node[child_key][:])
        elif child_key.startswith(_ARG_ARRAY_PREFIX):
            args[child_key[len(_ARG_ARRAY_PREFIX):]] = torch.from_numpy(node[child_key][:])
    spec[SPEC_ARGS_KEY] = args
    return spec, state_dict
