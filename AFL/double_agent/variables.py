"""Placeholders for dataset variables inside ordinary Python callables.

A pipeline is written before the data it will run on exists. Ops have always
bridged that gap by taking *names* of dataset variables in their constructors.
:class:`Variable` extends the same idea to arbitrary Python code: it is a
placeholder for a dataset variable that an ordinary function, lambda or
``functools.partial`` can close over::

    model = Variable("phase_model")          # a model ref written by an extrapolator
    composition = Variable("composition_next")

    op = SRSFHilbertSphereAcquisitionFunction(
        manifold=manifold,
        acquisition=lambda q: profile_score(q, composition, model, manifold, BOUNDS_T),
    )

While an op calculates it binds the dataset with :func:`bind_dataset`; inside
that block every ``Variable`` resolves to the variable's value and behaves like
it: attribute access, indexing, calls, arithmetic and ``torch.*`` functions are
all forwarded to the resolved value. ``variable.value`` returns the resolved
object itself, which is needed for ``isinstance`` checks and for tensor factories
such as ``torch.as_tensor`` that do not dispatch on proxies.

Resolution rules (``kind="auto"``):

* a 0-d string variable holding a model ref (see :mod:`AFL.double_agent.torch_io`)
  resolves to the loaded model,
* numeric or boolean data resolves to a ``torch.Tensor`` (default dtype),
* anything else resolves to a NumPy array,
* a name that is not in the dataset but was *provided* by the op (for example the
  upstream ``"model"`` and ``"best_f"`` that ``BoTorchAcquisition`` loads, or the resolved
  ``"manifold"`` of the SRSF op) resolves to that object unchanged.

``Variable.as_model()``, ``as_tensor()``, ``as_array()`` and ``raw()`` pin the
conversion explicitly.

:func:`find_variables` discovers the ``Variable`` placeholders a callable depends
on (closure cells, defaults, referenced globals, ``partial`` arguments and nested
containers) so ops can register them as inputs without the user repeating them.
"""

from __future__ import annotations

import contextvars
import functools
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import torch
import xarray as xr

VARIABLE_KEY = "variable"
REF_KEY = "ref"
KINDS = ("auto", "model", "tensor", "array", "raw")


class UnboundVariableError(RuntimeError):
    """A :class:`Variable` was used outside of :func:`bind_dataset`."""


class UnresolvedReferenceError(ValueError):
    """A pipeline JSON refers to live Python objects that were not supplied."""


@dataclass
class Binding:
    dataset: xr.Dataset
    provided: Dict[str, Any] = field(default_factory=dict)


_BINDING: contextvars.ContextVar[Optional[Binding]] = contextvars.ContextVar("afl_dataset_binding", default=None)


@contextmanager
def bind_dataset(dataset: xr.Dataset, provided: Optional[Mapping[str, Any]] = None) -> Iterator[Binding]:
    """Make ``dataset`` (and op-provided objects) visible to every :class:`Variable`.

    ``provided`` maps names to objects the op hands out itself (e.g. ``"model"``);
    ``"dataset"`` is always provided. Ops wrap the body of ``calculate`` in this; user
    code never needs to call it except to evaluate a score outside of a pipeline,
    e.g. for plotting.
    """
    binding = Binding(dataset=dataset, provided={"dataset": dataset, **(provided or {})})
    token = _BINDING.set(binding)
    try:
        yield binding
    finally:
        _BINDING.reset(token)


def current_binding() -> Binding:
    binding = _BINDING.get()
    if binding is None:
        raise UnboundVariableError(
            "No dataset is bound. Variables resolve only while a PipelineOp is calculating; "
            "wrap the call in `with bind_dataset(dataset): ...` to use them outside a pipeline."
        )
    return binding


def _convert(data: xr.DataArray, kind: str, device, dtype):
    from AFL.double_agent.torch_io import is_model_ref, load_model

    if kind == "raw":
        return data
    is_ref = data.ndim == 0 and is_model_ref(str(data.values[()]))
    if kind == "model" or (kind == "auto" and is_ref):
        if not is_ref:
            raise TypeError(f"Variable {data.name!r} is not a model ref: {data.values!r}")
        return load_model(str(data.values[()]))
    numeric = np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_)
    if kind == "tensor" or (kind == "auto" and numeric):
        return torch.as_tensor(np.asarray(data.values), dtype=dtype or torch.get_default_dtype(), device=device)
    return np.asarray(data.values)


def resolve_name(name: str, kind: str = "auto", device=None, dtype=None, binding: Optional[Binding] = None):
    """Resolve a variable name against the current (or given) binding."""
    binding = binding or current_binding()
    if name in binding.dataset:
        return _convert(binding.dataset[name], kind, device, dtype)
    if name in binding.provided:
        return binding.provided[name]
    available = sorted(binding.dataset.data_vars) + sorted(binding.provided)
    raise KeyError(f"Variable {name!r} is neither in the dataset nor provided by the op. Available: {available}")


class Variable:
    """Placeholder for a dataset variable; see the module docstring."""

    __slots__ = ("name", "kind", "device", "dtype")

    def __init__(self, name: str, kind: str = "auto", device=None, dtype=None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", dtype)

    # -- explicit conversions -------------------------------------------------
    def as_model(self) -> "Variable":
        return Variable(self.name, "model")

    def as_tensor(self, device=None, dtype=None) -> "Variable":
        return Variable(self.name, "tensor", device=device, dtype=dtype)

    def as_array(self) -> "Variable":
        return Variable(self.name, "array")

    def raw(self) -> "Variable":
        """Resolve to the ``xarray.DataArray`` itself."""
        return Variable(self.name, "raw")

    # -- resolution -------------------------------------------------------------
    @property
    def value(self):
        """The resolved object (requires an active :func:`bind_dataset`)."""
        return resolve_name(self.name, self.kind, self.device, self.dtype)

    # -- JSON ---------------------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {VARIABLE_KEY: self.name}
        if self.kind != "auto":
            out["kind"] = self.kind
        if self.device is not None:
            out["device"] = str(self.device)
        if self.dtype is not None:
            out["dtype"] = str(self.dtype).replace("torch.", "")
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Variable":
        dtype = data.get("dtype")
        return cls(
            data[VARIABLE_KEY],
            kind=data.get("kind", "auto"),
            device=data.get("device"),
            dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype,
        )

    # -- proxy behaviour --------------------------------------------------------
    def __repr__(self) -> str:
        extra = "" if self.kind == "auto" else f", kind={self.kind!r}"
        return f"Variable({self.name!r}{extra})"

    def __getattr__(self, name: str):
        if name.startswith("__") or name in Variable.__slots__:
            raise AttributeError(name)
        return getattr(self.value, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Variable placeholders are immutable")

    def __copy__(self) -> "Variable":
        return self

    def __deepcopy__(self, memo) -> "Variable":
        return self

    def __reduce__(self):
        return (Variable, (self.name, self.kind, self.device, self.dtype))

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

    def __getitem__(self, item):
        return self.value[item]

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __array__(self, dtype=None, copy=None):
        value = self.value
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    @classmethod
    def __torch_function__(cls, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        return func(*resolve_variables(args), **resolve_variables(kwargs))

    def _binary(op):  # noqa: N805 - helper, not a method
        def forward(self, other):
            return op(self.value, resolve_variables(other))

        def reverse(self, other):
            return op(resolve_variables(other), self.value)

        return forward, reverse

    __add__, __radd__ = _binary(lambda a, b: a + b)
    __sub__, __rsub__ = _binary(lambda a, b: a - b)
    __mul__, __rmul__ = _binary(lambda a, b: a * b)
    __truediv__, __rtruediv__ = _binary(lambda a, b: a / b)
    __pow__, __rpow__ = _binary(lambda a, b: a**b)
    __matmul__, __rmatmul__ = _binary(lambda a, b: a @ b)
    del _binary

    def __neg__(self):
        return -self.value


def is_variable_spec(value: Any) -> bool:
    return isinstance(value, dict) and VARIABLE_KEY in value


def is_ref(value: Any) -> bool:
    return isinstance(value, dict) and REF_KEY in value


def resolve_variables(obj: Any) -> Any:
    """Replace every :class:`Variable` inside ``obj`` (dict/list/tuple/scalar) with its value.

    ``{"variable": name}`` dicts (the JSON form a PipelineOp stores) are resolved too.
    """
    if isinstance(obj, Variable):
        return obj.value
    if is_variable_spec(obj):
        return Variable.from_json(obj).value
    if isinstance(obj, dict):
        return {k: resolve_variables(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(resolve_variables(v) for v in obj)
    return obj


def decode_variables(obj: Any) -> Any:
    """Turn ``{"variable": name, ...}`` dicts inside a JSON structure back into :class:`Variable`."""
    if is_variable_spec(obj):
        return Variable.from_json(obj)
    if isinstance(obj, dict):
        return {k: decode_variables(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_variables(v) for v in obj]
    return obj


def find_variables(obj: Any, _seen: Optional[set] = None, _depth: int = 0) -> Dict[str, Variable]:
    """Collect the :class:`Variable` placeholders reachable from ``obj``.

    Looks into functions (closure cells, defaults, globals they reference),
    ``functools.partial`` objects, bound methods and dict/list/tuple/set containers.
    Returns ``{name: Variable}`` in discovery order.
    """
    found: Dict[str, Variable] = {}
    seen = _seen if _seen is not None else set()
    if _depth > 8 or id(obj) in seen:
        return found
    seen.add(id(obj))

    def visit(child):
        found.update(find_variables(child, seen, _depth + 1))

    if isinstance(obj, Variable):
        found.setdefault(obj.name, obj)
    elif is_variable_spec(obj):
        found.setdefault(obj[VARIABLE_KEY], Variable.from_json(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            visit(v)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            visit(v)
    elif isinstance(obj, functools.partial):
        visit(obj.func)
        visit(obj.args)
        visit(obj.keywords)
    elif isinstance(obj, types.MethodType):
        visit(obj.__self__)
        visit(obj.__func__)
    elif isinstance(obj, types.FunctionType):
        for cell in obj.__closure__ or ():
            try:
                visit(cell.cell_contents)
            except ValueError:  # empty cell
                pass
        visit(obj.__defaults__ or ())
        visit(obj.__kwdefaults__ or {})
        for name in _referenced_globals(obj.__code__):
            if name in obj.__globals__:
                visit(obj.__globals__[name])
    return found


def _referenced_globals(code: types.CodeType) -> set:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _referenced_globals(const)
    return names
