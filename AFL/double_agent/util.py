"""
A collection of helper methods/classes
"""

import inspect
from typing import Any, Dict

from AFL.double_agent.PipelineOp import PipelineOp


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
