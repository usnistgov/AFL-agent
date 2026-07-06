"""
A collection of helper methods/classes
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Literal, Sequence

import numpy as np
import torch
import xarray as xr

from AFL.double_agent.PipelineOp import PipelineOp


ObjectiveDirection = Literal["maximize", "minimize"]


class ILRTransform(torch.nn.Module):
    r"""Isometric log-ratio transform used for simplex-aware Gaussian processes.

    The transform maps each composition :math:`x \in S^D` into orthonormal
    Euclidean coordinates via

    .. math::
        \operatorname{ilr}(x) = \operatorname{clr}(x) V,

    where :math:`V \in \mathbb{R}^{D \times (D-1)}` is a Helmert basis for the
    clr hyperplane and

    .. math::
        \operatorname{clr}(x)_i = \log(x_i) - \frac{1}{D}\sum_{j=1}^{D}\log(x_j).

    The Gaussian process is then fit with a Mat\'ern-$\nu=2.5$ kernel with
    automatic relevance determination in the transformed coordinates:

    .. math::
        k_S(x, x') = k_{\mathrm{Mat\'ern}}\left(\operatorname{ilr}(x), \operatorname{ilr}(x')\right).

    This preserves Aitchison geometry while using the simplex's intrinsic
    :math:`D-1` dimensional Euclidean representation.
    """

    def __init__(self, n_dim: int, eps: float = 1e-12) -> None:
        super().__init__()
        if n_dim < 2:
            raise ValueError("ILRTransform requires at least two simplex components.")
        self.eps = eps
        self.register_buffer("basis", self._helmert_submatrix(n_dim))

    @staticmethod
    def _helmert_submatrix(n_dim: int) -> torch.Tensor:
        basis = torch.zeros((n_dim, n_dim - 1), dtype=torch.double)
        for column in range(n_dim - 1):
            scale = torch.sqrt(torch.tensor((column + 1) * (column + 2), dtype=torch.double))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=self.eps)
        log_x = torch.log(x)
        clr = log_x - log_x.mean(dim=-1, keepdim=True)
        return clr @ self.basis


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
        from botorch.acquisition import LogExpectedImprovement, PosteriorMean, qLogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise ImportError(
            "BoTorch support requires the optional botorch dependencies. Install with 'pip install -e .[botorch]'."
        ) from exc

    return {
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "LogExpectedImprovement": LogExpectedImprovement,
        "MaternKernel": MaternKernel,
        "PosteriorMean": PosteriorMean,
        "ScaleKernel": ScaleKernel,
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


def make_simplex_constraints(n_dim: int) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    r"""Return BoTorch-compatible simplex constraints.

    BoTorch's `optimize_acqf` accepts linear inequality constraints of the form

    .. math::
        \sum_i a_i x_i \ge b.

    A simplex equality :math:`\sum_i x_i = 1` is therefore encoded as the pair

    .. math::
        \sum_i x_i \ge 1, \qquad -\sum_i x_i \ge -1,

    which together are exactly equivalent to the equality constraint. This helper
    returns that pair so the optimizer can enforce the simplex using the API it
    already exposes.
    """
    indices = torch.arange(n_dim, dtype=torch.long)
    coefficients = torch.ones(n_dim, dtype=torch.double)
    return [(indices, coefficients, 1.0), (indices, -coefficients, -1.0)]


def sample_dirichlet_initial_conditions(
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    n_dim = bounds.shape[-1]
    concentration = torch.full((n_dim,), alpha, dtype=bounds.dtype, device=bounds.device)
    distribution = torch.distributions.Dirichlet(concentration)
    return distribution.sample((num_restarts, q))


def fit_single_task_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    standardize: bool = True,
    is_simplex: bool = False,
):
    botorch = import_botorch()
    outcome_transform = botorch["Standardize"](m=train_y.shape[-1]) if standardize else None
    simplex_transform = ILRTransform(train_x.shape[-1]) if is_simplex else None
    transformed_x = simplex_transform(train_x) if simplex_transform is not None else train_x
    model_kwargs: dict[str, Any] = {
        "train_X": transformed_x,
        "train_Y": train_y,
        "outcome_transform": outcome_transform,
    }

    if is_simplex:
        ard_num_dims = transformed_x.shape[-1]
        model_kwargs["covar_module"] = botorch["ScaleKernel"](
            botorch["MaternKernel"](nu=2.5, ard_num_dims=ard_num_dims)
        )

    model = botorch["SingleTaskGP"](**model_kwargs)
    mll = botorch["ExactMarginalLogLikelihood"](model.likelihood, model)
    botorch["fit_gpytorch_mll"](mll)
    model._simplex_input_transform = simplex_transform
    model._is_simplex = is_simplex
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
    if mean.shape[0] != grid_index.shape[0] or variance.shape[0] != grid_index.shape[0]:
        raise ValueError(
            "Posterior outputs must match the candidate grid length: "
            f"got mean={mean.shape[0]}, variance={variance.shape[0]}, grid={grid_index.shape[0]}."
        )
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


def _transform_model_inputs(model, x: torch.Tensor) -> torch.Tensor:
    transform = getattr(model, "_simplex_input_transform", None)
    if transform is None:
        return x
    return transform(x)


def optimize_posterior_mean(
    model,
    bounds: torch.Tensor,
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 128,
    objective_direction: ObjectiveDirection = "maximize",
    inequality_constraints: Sequence[tuple[torch.Tensor, torch.Tensor, float]] | None = None,
    batch_initial_conditions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    botorch = import_botorch()

    class _TransformedPosteriorMean(torch.nn.Module):
        def __init__(self, gp_model) -> None:
            super().__init__()
            self.posterior_mean = botorch["PosteriorMean"](gp_model)
            self.gp_model = gp_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.posterior_mean(_transform_model_inputs(self.gp_model, x))

    if batch_initial_conditions is None and inequality_constraints is not None:
        batch_initial_conditions = sample_dirichlet_initial_conditions(
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
        )

    candidates, values = botorch["optimize_acqf"](
        acq_function=_TransformedPosteriorMean(model),
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=inequality_constraints,
        batch_initial_conditions=batch_initial_conditions,
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
    transformed_x = _transform_model_inputs(model, candidate_x)
    return acquisition(transformed_x.unsqueeze(-2)).detach()


def evaluate_qlog_expected_improvement(
    model,
    candidate_x: torch.Tensor,
    best_f: float,
    q: int,
) -> torch.Tensor:
    botorch = import_botorch()
    acquisition = botorch["qLogExpectedImprovement"](model=model, best_f=best_f)
    transformed_x = _transform_model_inputs(model, candidate_x)
    if q <= 1:
        return acquisition(transformed_x.unsqueeze(-2)).detach()

    values = []
    total = transformed_x.shape[0]
    for start in range(0, total - q + 1):
        batch = transformed_x[start : start + q]
        values.append(acquisition(batch.unsqueeze(0)).reshape(()))

    padded = torch.full((total,), float("-inf"), dtype=transformed_x.dtype)
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
    inequality_constraints: Sequence[tuple[torch.Tensor, torch.Tensor, float]] | None = None,
    batch_initial_conditions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    botorch = import_botorch()
    bounds = bounds_from_tensor(candidate_x)

    class _TransformedAcquisition(torch.nn.Module):
        def __init__(self, base_acquisition, gp_model) -> None:
            super().__init__()
            self.base_acquisition = base_acquisition
            self.gp_model = gp_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base_acquisition(_transform_model_inputs(self.gp_model, x))

    if acquisition_kind == "qlogei":
        base_acq_function = botorch["qLogExpectedImprovement"](model=model, best_f=best_f)
    else:
        base_acq_function = botorch["LogExpectedImprovement"](model=model, best_f=best_f)
    acq_function = _TransformedAcquisition(base_acq_function, model)

    if batch_initial_conditions is None and inequality_constraints is not None:
        batch_initial_conditions = sample_dirichlet_initial_conditions(
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
        )

    candidates, values = botorch["optimize_acqf"](
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=inequality_constraints,
        batch_initial_conditions=batch_initial_conditions,
    )

    return candidates.detach(), values.detach()
