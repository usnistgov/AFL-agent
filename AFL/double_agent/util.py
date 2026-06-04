"""
A collection of helper methods/classes
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Literal

import numpy as np
import torch
import xarray as xr

from AFL.double_agent.PipelineOp import PipelineOp


ObjectiveDirection = Literal["maximize", "minimize"]


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


def import_botorch():
    try:
        from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise ImportError(
            "BoTorch support requires the optional botorch dependencies. Install with 'pip install -e .[botorch]'."
        ) from exc

    return {
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "LogExpectedImprovement": LogExpectedImprovement,
        "SingleTaskGP": SingleTaskGP,
        "Standardize": Standardize,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "optimize_acqf": optimize_acqf,
        "qLogExpectedImprovement": qLogExpectedImprovement,
    }


def _as_float_tensor(data: xr.DataArray, leading_dim: str) -> torch.Tensor:
    values = data.transpose(leading_dim, ...).values
    return torch.as_tensor(np.asarray(values), dtype=torch.double)


def dataset_to_botorch_training_data(
    dataset: xr.Dataset,
    feature_input_variable: str,
    predictor_input_variable: str,
    sample_dim: str,
    objective_direction: ObjectiveDirection = "minimize",
) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = _as_float_tensor(dataset[feature_input_variable], sample_dim)
    train_y = _as_float_tensor(dataset[predictor_input_variable], sample_dim)

    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)

    if objective_direction == "minimize":
        train_y = -train_y

    return train_x, train_y


def dataset_to_botorch_candidates(
    dataset: xr.Dataset,
    grid_variable: str,
    grid_dim: str,
) -> torch.Tensor:
    return _as_float_tensor(dataset[grid_variable], grid_dim)


def fit_single_task_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    standardize: bool = True,
):
    botorch = import_botorch()
    outcome_transform = botorch["Standardize"](m=train_y.shape[-1]) if standardize else None
    model = botorch["SingleTaskGP"](
        train_X=train_x,
        train_Y=train_y,
        outcome_transform=outcome_transform,
    )
    mll = botorch["ExactMarginalLogLikelihood"](model.likelihood, model)
    botorch["fit_gpytorch_mll"](mll)
    return model


def posterior_to_xarray(
    posterior,
    grid_index: xr.DataArray,
    output_prefix: str,
    objective_direction: ObjectiveDirection = "maximize",
) -> xr.Dataset:
    mean = posterior.mean.detach().cpu().numpy().reshape(-1)
    if objective_direction == "minimize":
        mean = -mean
    variance = posterior.variance.detach().cpu().numpy().reshape(-1)
    return xr.Dataset(
        data_vars={
            f"{output_prefix}_mean": ((grid_index.dims[0],), mean),
            f"{output_prefix}_variance": ((grid_index.dims[0],), variance),
        },
        coords={grid_index.dims[0]: grid_index.values},
    )


def get_observed_best_f(
    train_y: torch.Tensor,
    objective_direction: ObjectiveDirection = "maximize",
) -> float:
    best_f = float(train_y.max().detach().cpu().item())
    if objective_direction == "minimize":
        return -best_f
    return best_f


def optimize_posterior_mean(
    model,
    bounds: torch.Tensor,
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 128,
    objective_direction: ObjectiveDirection = "maximize",
) -> tuple[torch.Tensor, float]:
    botorch = import_botorch()
    from botorch.acquisition import PosteriorMean

    candidates, values = botorch["optimize_acqf"](
        acq_function=PosteriorMean(model),
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    best_f = float(values.max().detach().cpu().item())
    if objective_direction == "minimize":
        best_f = -best_f
    return candidates.detach(), best_f


def bounds_from_tensor(points: torch.Tensor) -> torch.Tensor:
    lower = points.min(dim=0).values
    upper = points.max(dim=0).values
    return torch.stack([lower, upper])


def evaluate_log_expected_improvement(
    model,
    candidate_x: torch.Tensor,
    best_f: float,
) -> torch.Tensor:
    botorch = import_botorch()
    acquisition = botorch["LogExpectedImprovement"](model=model, best_f=best_f)
    return acquisition(candidate_x.unsqueeze(-2)).detach()


def evaluate_qlog_expected_improvement(
    model,
    candidate_x: torch.Tensor,
    best_f: float,
    q: int,
) -> torch.Tensor:
    botorch = import_botorch()
    acquisition = botorch["qLogExpectedImprovement"](model=model, best_f=best_f)
    if q <= 1:
        return acquisition(candidate_x.unsqueeze(-2)).detach()

    values = []
    total = candidate_x.shape[0]
    for start in range(0, total - q + 1):
        batch = candidate_x[start : start + q]
        values.append(acquisition(batch.unsqueeze(0)).reshape(()))

    padded = torch.full((total,), float("-inf"), dtype=candidate_x.dtype)
    if values:
        padded[: len(values)] = torch.stack(values)
    return padded


def optimize_acquisition_function(
    model,
    candidate_x: torch.Tensor,
    best_f: float,
    acquisition_kind: str = "logei",
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    botorch = import_botorch()
    bounds = bounds_from_tensor(candidate_x)

    if acquisition_kind == "qlogei":
        acq_function = botorch["qLogExpectedImprovement"](model=model, best_f=best_f)
    else:
        acq_function = botorch["LogExpectedImprovement"](model=model, best_f=best_f)

    candidates, values = botorch["optimize_acqf"](
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return candidates.detach(), values.detach()
