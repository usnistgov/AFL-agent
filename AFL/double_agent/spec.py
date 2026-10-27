"""JSON-serializable descriptions of Python classes.

A *class spec* is a dictionary of the form::

    {"class": "package.module.ClassName", "args": {...}}

``args`` is optional. Specs let a :class:`~AFL.double_agent.PipelineOp.PipelineOp`
accept Python classes as constructor arguments while remaining JSON
serializable: :func:`encode_spec` is applied to every constructor argument when
an op is built, and ops call :func:`resolve_spec` inside ``calculate`` to build
the actual object.

A *function spec* refers to a plain Python function that an op should call
rather than instantiate::

    {"function": "package.module.function_name"}

:func:`encode_spec` produces one from any module-level function and
:func:`resolve_function` imports it back. Functions defined in ``__main__``
(a notebook or script) are accepted: they resolve within the same process, but
a pipeline saved from that process cannot be reloaded elsewhere.

Values that have no JSON form at all (lambdas, closures, live instances) are kept
in memory by the op and exported as *references*, ``{"ref": "<op name>.<arg>"}``;
see :class:`~AFL.double_agent.PipelineOp.PipelineOp` for how they are supplied
again when a pipeline is loaded. :class:`~AFL.double_agent.variables.Variable`
placeholders encode as ``{"variable": name}``.
"""

from __future__ import annotations

import importlib
import types
from typing import Any, Callable, Dict, Union

SPEC_CLASS_KEY = "class"
SPEC_ARGS_KEY = "args"
SPEC_FUNCTION_KEY = "function"

Spec = Dict[str, Any]


def class_path(klass: Callable) -> str:
    """Return the fully qualified import path of a class (or module-level function).

    Raises
    ------
    TypeError
        If the object was defined in ``__main__`` (notebook or script) or is a
        local/lambda, and so could not be re-imported by :func:`resolve_spec`.
    """
    module = getattr(klass, "__module__", None)
    qualname = getattr(klass, "__qualname__", None)
    if module in (None, "__main__", "builtins") or qualname is None or "<" in qualname:
        raise TypeError(
            f"{klass!r} is not importable by path (module={module!r}, qualname={qualname!r}). "
            "Define it at module level in an importable module to use it in a PipelineOp."
        )
    return f"{module}.{qualname}"


def function_path(function: Callable) -> str:
    """Return the import path of a module-level function.

    Unlike :func:`class_path`, functions defined in ``__main__`` are allowed because
    ``importlib.import_module("__main__")`` resolves them within the same process.

    Raises
    ------
    TypeError
        If the object is a lambda, a local/nested function, or has no module.
    """
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    if module in (None, "builtins") or qualname is None or "<" in qualname:
        raise TypeError(
            f"{function!r} is not importable by path (module={module!r}, qualname={qualname!r}). "
            "Lambdas and functions defined inside another function cannot be used in a PipelineOp; "
            "define the function at module level."
        )
    return f"{module}.{qualname}"


def is_spec(value: Any) -> bool:
    """True if ``value`` is a dict with a ``"class"`` key."""
    return isinstance(value, dict) and SPEC_CLASS_KEY in value


def is_function_spec(value: Any) -> bool:
    """True if ``value`` is a dict with a ``"function"`` key."""
    return isinstance(value, dict) and SPEC_FUNCTION_KEY in value


def encode_spec(value: Any) -> Any:
    """Convert Python classes and functions to JSON-safe specs; return other values unchanged.

    * ``type`` -> ``{"class": "mod.Name"}``
    * module-level function -> ``{"function": "mod.name"}``
    * ``{"class": SomeClass, "args": {...}}`` -> ``{"class": "mod.Name", "args": {...}}``
    * ``{"function": some_function}`` -> ``{"function": "mod.name"}``
    * ``Variable`` placeholder -> ``{"variable": name}``
    * anything else -> unchanged
    """
    from AFL.double_agent.variables import Variable

    if isinstance(value, Variable):
        return value.to_json()
    if isinstance(value, type):
        return {SPEC_CLASS_KEY: class_path(value)}
    if isinstance(value, types.FunctionType):
        return {SPEC_FUNCTION_KEY: function_path(value)}
    if is_function_spec(value) and callable(value[SPEC_FUNCTION_KEY]):
        encoded = dict(value)
        encoded[SPEC_FUNCTION_KEY] = function_path(value[SPEC_FUNCTION_KEY])
        return encoded
    if is_spec(value) and isinstance(value[SPEC_CLASS_KEY], type):
        encoded = dict(value)
        encoded[SPEC_CLASS_KEY] = class_path(value[SPEC_CLASS_KEY])
        return encoded
    return value


def import_class(path: str) -> Callable:
    """Import ``"package.module.Name"`` (or ``"module.Outer.Inner"``) and return the class.

    Module-level factory functions are accepted as well; anything callable qualifies.
    """
    module_name, _, qualname = path.rpartition(".")
    # Walk back through the dotted path until something imports, so that nested
    # classes ("module.Outer.Inner") resolve as well.
    obj = None
    attrs = [qualname]
    while module_name:
        try:
            obj = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            module_name, _, attr = module_name.rpartition(".")
            attrs.insert(0, attr)
    if obj is None:
        raise ImportError(f"Could not import any module on the path {path!r}")
    for attr in attrs:
        obj = getattr(obj, attr)
    if not callable(obj):
        raise TypeError(f"{path!r} resolved to {obj!r}, which is not a class or callable")
    return obj


def resolve_spec(spec: Union[Spec, str, Callable], **override_args: Any) -> Any:
    """Build an instance from a spec, import path string, or class.

    Parameters
    ----------
    spec
        ``{"class": "mod.Name", "args": {...}}``, a bare import path string
        (no constructor args), or a class/factory object. ``"class"`` may also
        name a module-level factory function.
    **override_args
        Extra keyword arguments passed to the constructor, overriding any in
        the spec. Useful for arguments that only exist at calculate time.
    """
    if isinstance(spec, type) or (callable(spec) and not isinstance(spec, dict)):
        klass, args = spec, {}
    elif isinstance(spec, str):
        klass, args = import_class(spec), {}
    elif is_spec(spec):
        klass = spec[SPEC_CLASS_KEY]
        if isinstance(klass, str):
            klass = import_class(klass)
        args = dict(spec.get(SPEC_ARGS_KEY) or {})
    else:
        raise TypeError(
            f"Expected a spec dict, import path string, or class; got {type(spec).__name__}: {spec!r}"
        )
    args.update(override_args)
    return klass(**args)


def resolve_function(spec: Union[Dict[str, Any], str, Callable]) -> Callable:
    """Return the callable described by a function spec without calling it.

    Parameters
    ----------
    spec
        ``{"function": "mod.name"}``, a bare import path string, or a callable
        (returned unchanged).
    """
    if callable(spec) and not isinstance(spec, dict):
        return spec
    if isinstance(spec, str):
        return import_class(spec)
    if is_function_spec(spec):
        target = spec[SPEC_FUNCTION_KEY]
        return target if callable(target) else import_class(target)
    raise TypeError(
        f"Expected a function spec dict, import path string, or callable; got {type(spec).__name__}: {spec!r}"
    )
