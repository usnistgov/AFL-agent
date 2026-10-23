"""
A collection of helper methods/classes
"""

from __future__ import annotations

import inspect
import logging
from typing import Dict

from AFL.double_agent.PipelineOp import PipelineOp


logger = logging.getLogger(__name__)


def listify(obj):
    """Make any input an iterable list

    The primary use case is to handle inputs that are sometimes length=1 and not always passed as lists. In particular,
    this method handles string inputs which we do not want to iterate over.

    Example
    -------
    ```python
    def my_func(input):
        for i in listify(input):
            print(i)

    In[1]: my_func(1)
    Out[2]:
    1

    In[1]: my_func([1,2])
    Out[2]:
    1
    2

    In[1]: my_func('test')
    Out[2]:
    'test'
    ```

    In the last example, without listify the result would have been t,e,s,t on newlines.
    """
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        obj = [obj]
    return obj


def extract_parameters(op: PipelineOp, method: str = "__init__") -> Dict:
    """Attempt to reconstruct the input parameters for a object's constructor

    Parameters
    ----------
    op: Any
        Technically any Python object but targeted at PipelineOps

    method: str
        While method to try to reconstruct. Typically, __init__
    """
    # grab base signature and default parameters
    signature = inspect.signature(getattr(op, method))

    parameters = {k: v.default for k, v in signature.parameters.items()}
    for k, v in signature.parameters.items():
        if k in ('input_variables','output_variables') and k not in op.__dict__:
            parameters[k] = op.__dict__.get(k[:-1], v)
        else:
            parameters[k] = op.__dict__.get(k, v)

    return parameters


# The BoTorch helpers used to live here. They moved to ``AFL.double_agent.BoTorchMixin``;
# the shim below keeps old imports working for one release.
_MOVED_TO_BOTORCH_MIXIN = {
    "ObjectiveDirection",
    "ScoreFunctionAcquisition",
    "build_acquisition",
    "build_single_task_gp",
    "dataset_to_botorch_candidates",
    "dataset_to_botorch_training_data",
    "evaluate_acquisition_on_grid",
    "fit_single_task_gp",
    "get_observed_best_f",
    "import_botorch",
    "make_simplex_constraints",
    "optimization_bounds_to_tensor",
    "optimization_bounds_to_xarray",
    "posterior_to_xarray",
}


def __getattr__(name: str):
    if name in _MOVED_TO_BOTORCH_MIXIN:
        import importlib
        import warnings

        module = importlib.import_module("AFL.double_agent.BoTorchMixin")
        if hasattr(module, name):  # ScoreFunctionAcquisition, ObjectiveDirection
            target, where = getattr(module, name), f"AFL.double_agent.BoTorchMixin.{name}"
        else:
            attr = "build_builtin_acquisition" if name == "build_acquisition" else name
            target, where = getattr(module.BoTorchMixin, attr), f"AFL.double_agent.BoTorchMixin.BoTorchMixin.{attr}"
        warnings.warn(f"AFL.double_agent.util.{name} moved to {where}", DeprecationWarning, stacklevel=2)
        return target
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
