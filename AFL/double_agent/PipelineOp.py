import copy
import re
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import inspect
import json
import importlib

import matplotlib.pyplot as plt
import xarray as xr
from typing_extensions import Self

from AFL.double_agent.PipelineContext import PipelineContext, NoContextException
from AFL.double_agent.spec import encode_spec
from AFL.double_agent.variables import (
    REF_KEY,
    UnresolvedReferenceError,
    decode_variables,
    find_variables,
    is_ref,
)


def _encode_deep(value):
    """Apply :func:`encode_spec` through dicts and lists."""
    value = encode_spec(value)
    if isinstance(value, dict):
        return {k: _encode_deep(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_deep(v) for v in value]
    return value


_NOT_JSON = object()  # sentinel: value has no JSON form and is kept live


def _json_safe(value):
    """Encode ``value`` (recursively) if the result is JSON serializable, else ``_NOT_JSON``."""
    try:
        encoded = _encode_deep(value)
        json.dumps(encoded)
    except (TypeError, OverflowError, ValueError):
        return _NOT_JSON
    return encoded


class PipelineOp(ABC):
    """Abstract base class for data processors. All operations in AFL.double_agent should inherit PipelineOpBase.

    Parameters
    ----------
    name : Optional[str] | List[str]
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    input_variable : Optional[str] | List[str]
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : Optional[str] | List[str]
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    input_prefix : Optional[str] | List[str]
        Prefix for input variables when using pattern matching
    output_prefix : Optional[str] | List[str]
        Prefix for output variables when using pattern matching
    """

    def __init__(
        self,
        name: Optional[str] | List[str] = None,
        input_variable: Optional[str] | List[str] = None,
        output_variable: Optional[str] | List[str] = None,
        input_prefix: Optional[str] | List[str] = None,
        output_prefix: Optional[str] | List[str] = None,
    ):

        if all(
            x is None
            for x in [input_variable, output_variable, input_prefix, output_prefix]
        ):
            warnings.warn(
                "No input/output information set for PipelineOp...this is likely an error",
                stacklevel=2,
            )


        self.name = name or "PipelineOp"
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.output: Dict[str, xr.DataArray] = {}

        # variables to exclude when constructing attrs dict for xarray
        self._banned_from_attrs = ["output", "_banned_from_attrs"]

        ## PipelineContext
        try:
            # try to add this object to current pipeline on context stack
            PipelineContext.get_context().append(self)
        except NoContextException:
            # silently continue for those working outside a context manager
            pass


        ## Gathering Arguments of Most-derived Child Class
        # Retrieve the full stack.
        stack = inspect.stack()
        valid_frames = []
        # Iterate through all frames in the stack.
        for frame_info in stack:
            # Look for __init__ functions where 'self' is our instance.
            if (
                frame_info.function == "__init__"
                and frame_info.frame.f_locals.get("self") is self
            ):
                valid_frames.append(frame_info)
        if valid_frames:
            # Choose the last __init__ call in the inheritance chain.
            final_frame = valid_frames[-1].frame
            args_info = inspect.getargvalues(final_frame)
        else:
            # Fallback: use the immediate caller if nothing was found.
            final_frame = inspect.currentframe().f_back
            args_info = inspect.getargvalues(final_frame)

        # Build _stored_args from the constructor arguments. Values with a JSON form are
        # stored encoded (classes and functions become import-path specs, see
        # AFL.double_agent.spec). Anything else (lambdas, closures, live instances) is
        # stored as the live object and exported by to_json() as a reference.
        stored_args = {}
        for arg in args_info.args:
            if arg != "self":
                raw = args_info.locals[arg]
                encoded = _json_safe(raw)
                stored_args[arg] = raw if encoded is _NOT_JSON else encoded
        self._stored_args = stored_args
        self._register_variable_inputs()

    # Names that ops hand to Variables themselves (not dataset variables); subclasses override.
    PROVIDED_KEYS: tuple = ()

    def _register_variable_inputs(self) -> None:
        """Add dataset variables referenced by ``Variable`` placeholders to ``input_variable``."""
        names = [
            n for n in find_variables(object.__getattribute__(self, "_stored_args"))
            if n not in self.PROVIDED_KEYS
        ]
        if not names:
            return
        current = self.input_variable
        if current is None:
            current = []
        elif isinstance(current, str):
            current = [current]
        else:
            current = list(current)
        # Goes through __setattr__ so a stored "input_variable" constructor argument is updated too.
        self.input_variable = current + [n for n in names if n not in current]

    def live_objects(self) -> Dict[str, object]:
        """Constructor arguments kept in memory (no JSON form), keyed by their export reference."""
        return {
            f"{self.name}.{arg}": value
            for arg, value in object.__getattribute__(self, "_stored_args").items()
            if _json_safe(value) is _NOT_JSON
        }

    def __getattribute__(self, name):
        if name == "_stored_args":
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                return {}
        try:
            stored_args = object.__getattribute__(self, "_stored_args")
        except AttributeError:
            return object.__getattribute__(self, name)
        if name in stored_args:
            return stored_args[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "_stored_args":
            return object.__setattr__(self, name, value)

        try:
            stored_args = object.__getattribute__(self, "_stored_args")
        except AttributeError:
            stored_args = None

        if stored_args is not None and name in stored_args:
            encoded = _json_safe(value)
            stored_args[name] = value if encoded is _NOT_JSON else encoded
        else:
            object.__setattr__(self, name, value)

    def to_json(self):
        """Serialize the class path and constructor arguments to a JSON-safe dict.

        Arguments without a JSON form are written as ``{"ref": "<name>.<arg>"}``; the
        live objects are available from :meth:`live_objects` and must be passed back to
        :meth:`from_json` (``objects=``) when the pipeline is reloaded.
        """
        cls = self.__class__
        module = cls.__module__
        qualname = cls.__qualname__
        args = {}
        for arg, value in self._stored_args.items():
            encoded = _json_safe(value)
            if encoded is _NOT_JSON:
                encoded = {
                    REF_KEY: f"{self.name}.{arg}",
                    "type": type(value).__name__,
                    "variables": sorted(find_variables(value)),
                }
            args[arg] = encoded
        data = {
            "class": f"{module}.{qualname}",
            "args": args,
            "docstring": inspect.getdoc(cls) or "",
        }
        return data

    @classmethod
    def from_json(cls, json_data, objects: Optional[Dict[str, object]] = None):
        """Rebuild an op from :meth:`to_json` output.

        Parameters
        ----------
        json_data : dict
            ``{"class": ..., "args": {...}}``.
        objects : dict, optional
            Live objects for ``{"ref": ...}`` arguments, keyed by reference
            (``"<op name>.<arg>"``), e.g. ``pipeline.objects`` of the pipeline that was
            exported or :func:`AFL.double_agent.Pipeline.load_environment` of a module
            that defines it.

        Raises
        ------
        UnresolvedReferenceError
            If an argument is a reference and ``objects`` does not contain it.
        """
        fqcn = json_data["class"]  # e.g. "module.ClassName"
        args = json_data["args"]

        # Coerce string 'null' values to Python None (can happen from GUI inputs)
        args = cls._coerce_json_values(args)
        args = decode_variables(args)

        objects = objects or {}
        resolved = {}
        missing = []
        for arg, value in args.items():
            if is_ref(value):
                if value[REF_KEY] in objects:
                    resolved[arg] = objects[value[REF_KEY]]
                else:
                    missing.append(value[REF_KEY])
            else:
                resolved[arg] = value
        if missing:
            raise UnresolvedReferenceError(
                f"{fqcn} refers to live Python objects that were not supplied: {missing}. "
                "Pass them with `objects=` (e.g. `pipeline.objects` of the defining pipeline, or "
                "`load_environment('my_module')` for a module that builds it)."
            )

        mod_name, class_name = fqcn.rsplit(".", 1)
        module = importlib.import_module(mod_name)
        klass = getattr(module, class_name)
        return klass(**resolved)

    @staticmethod
    def _coerce_json_values(obj):
        """
        Recursively convert string representations of JSON primitives to their
        Python equivalents. This handles cases where 'null', 'true', 'false',
        or numeric strings come through from GUI text inputs.
        """
        if isinstance(obj, dict):
            return {k: PipelineOp._coerce_json_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [PipelineOp._coerce_json_values(item) for item in obj]
        elif isinstance(obj, str):
            stripped = obj.strip()
            # Handle null
            if stripped == 'null':
                return None
            # Handle booleans
            if stripped == 'true':
                return True
            if stripped == 'false':
                return False
            # Handle numeric values (integers and floats including scientific notation)
            # Use regex to match valid numeric patterns
            if re.match(r'^-?\d+$', stripped):
                # Integer
                return int(stripped)
            if re.match(r'^-?\d+\.?\d*(?:[eE][+-]?\d+)?$', stripped):
                # Float or scientific notation
                try:
                    return float(stripped)
                except ValueError:
                    pass
            return obj
        else:
            return obj

    @abstractmethod
    def calculate(self, dataset: xr.Dataset) -> Self:
        pass

    def __repr__(self) -> str:
        return f"<PipelineOp:{self.name}>"

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def _prefix_output(self, variable_name: str) -> str:
        prefixed_variable = copy.deepcopy(variable_name)
        if self.output_prefix is not None:
            prefixed_variable = f"{self.output_prefix}_{prefixed_variable}"
        return prefixed_variable

    def _get_attrs(self) -> Dict:
        # Shallow copy: every value is stringified below, and live constructor arguments
        # (virtual instruments, compiled kernels) may not be deep-copyable.
        output_dict = dict(self.__dict__)
        for key in self._banned_from_attrs:
            try:
                del output_dict[key]
            except KeyError:
                pass

        # sanitize
        for key in output_dict.keys():
            output_dict[key] = str(output_dict[key])
            # if output_dict[key] is None:
            #     output_dict[key] = 'None'
            # elif type(output_dict[key]) is bool:
            #     output_dict[key] = str(output_dict[key])

        return output_dict

    def _get_variable(self, dataset: xr.Dataset) -> xr.DataArray:
        if self.input_variable is None and self.input_prefix is None:
            raise ValueError(
                (
                    """Can't get variable for {self.name} without input_variable """
                    """or input_prefix specified in constructor """
                )
            )

        if self.input_variable is not None and self.input_prefix is not None:
            raise ValueError(
                (
                    """Both input_variable and input_prefix were specified in constructor. """
                    """Only one should be specified to avoid ambiguous operation"""
                )
            )

        if self.input_variable is not None:
            output = dataset[self.input_variable].copy()

        elif self.input_prefix is not None:
            raise NotImplementedError
        else:
            output = None

        return output

    def add_to_dataset(self, dataset, copy_dataset=True):
        """Adds (xarray) data in output dictionary to provided xarray dataset"""
        if copy_dataset:
            dataset1 = dataset.copy()
        else:
            dataset1 = dataset

        for name, value in self.output.items():
            if isinstance(value, xr.Dataset):
                # add PipelineOp meta variables to attributes
                for data_var in value:
                    value[data_var].attrs.update(self._get_attrs())
                dataset1 = xr.merge([dataset1, value])
            elif isinstance(value, xr.DataArray):
                # add PipelineOp meta variables to attributes
                value.attrs.update(self._get_attrs())
                dataset1[name] = value
            else:
                raise ValueError(
                    (
                        f"""Items in output dictionary of PipelineOp {self.name} must be xr.Dataset or xr.DataArray """
                        f"""Found variable named {name} of type {type(value)}."""
                    )
                )
        return dataset1

    def add_to_tiled(self, tiled_data):
        """Adds data in output dictionary to provided tiled catalogue"""
        raise NotImplementedError
        # This needs to handle/deconstruct xarray types!!
        # for name, dataarray in self.output.items():
        #     tiled_data.add_array(name, value.values)

    def plot(self, sample_dim: str = "sample", **mpl_kwargs) -> plt.Figure:
        """Plots the output of the PipelineOp.

        This method attempts to guess how to plot the data produced by the operation.
        """
        n = len(self.output)
        if n > 0:
            fig, axes = plt.subplots(n, 1, figsize=(6, n * 3))

            if n > 1:
                axes = list(axes.flatten())
            else:
                axes = [axes]

            for i, (name, data) in enumerate(self.output.items()):
                if data.ndim > 1 and (sample_dim in data.dims):
                    data.plot(hue=sample_dim, ax=axes[i], **mpl_kwargs)

                else:
                    data.plot(ax=axes[i], **mpl_kwargs)
                axes[i].set(title=name)
            return fig
        else:
            return plt.figure()
