"""
A collection of helper methods/classes
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Literal, Mapping, Sequence

import numpy as np
import torch
import xarray as xr

from AFL.double_agent.PipelineOp import PipelineOp


ObjectiveDirection = Literal["maximize", "minimize"]
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


def import_botorch():
    try:
        from botorch.acquisition import LogExpectedImprovement, PosteriorMean, qLogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.utils.sampling import draw_sobol_samples
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise ImportError(
            "BoTorch support requires the optional botorch dependencies. Install with 'pip install -e .[botorch]'."
        ) from exc

    return {
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "LogExpectedImprovement": LogExpectedImprovement,
        "PosteriorMean": PosteriorMean,
        "SingleTaskGP": SingleTaskGP,
        "Standardize": Standardize,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "optimize_acqf": optimize_acqf,
        "qLogExpectedImprovement": qLogExpectedImprovement,
        "draw_sobol_samples": draw_sobol_samples,
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


def optimization_bounds_to_tensor(
    bounds: Mapping[str, Mapping[str, float] | Sequence[float]] | Sequence[Sequence[float]],
    component_names: Sequence[Any] | None = None,
) -> torch.Tensor:
    """Convert explicit constructor bounds into a BoTorch bounds tensor.

    Parameters
    ----------
    bounds
        Either a mapping from component name to ``{"min": lower, "max": upper}``
        dictionaries, a mapping from component name to ``(lower, upper)`` pairs,
        or an ordered sequence of ``(lower, upper)`` pairs.
    component_names
        Optional ordered component names used to align mapping-based bounds with
        model inputs.
    """
    if isinstance(bounds, Mapping):
        if component_names is None:
            raise ValueError("component_names are required when bounds are provided as a mapping.")
        missing = [str(name) for name in component_names if name not in bounds]
        if missing:
            raise ValueError(f"Optimization bounds are missing components: {missing}")
        ordered_bounds = [bounds[name] for name in component_names]
    else:
        ordered_bounds = list(bounds)
        if component_names is not None and len(ordered_bounds) != len(component_names):
            raise ValueError(
                "Optimization bounds length does not match the number of components: "
                f"got {len(ordered_bounds)} bounds for {len(component_names)} components."
            )

    if not ordered_bounds:
        raise ValueError("Optimization bounds cannot be empty.")

    lower = []
    upper = []
    for index, pair in enumerate(ordered_bounds):
        if isinstance(pair, Mapping):
            if "min" not in pair or "max" not in pair:
                raise ValueError(
                    "Mapping-based optimization bounds must contain 'min' and 'max' keys: "
                    f"received {pair!r} at position {index}."
                )
            lo = float(pair["min"])
            hi = float(pair["max"])
        else:
            if len(pair) != 2:
                raise ValueError(
                    "Each optimization bound must contain exactly two values: "
                    f"received {pair!r} at position {index}."
                )
            lo = float(pair[0])
            hi = float(pair[1])
        if lo > hi:
            raise ValueError(
                "Optimization bounds must satisfy lower <= upper: "
                f"received ({lo}, {hi}) at position {index}."
            )
        lower.append(lo)
        upper.append(hi)

    return torch.tensor([lower, upper], dtype=torch.double)


def optimization_bounds_to_xarray(
    bounds: torch.Tensor,
    component_names: Sequence[Any],
    variable_name: str,
) -> xr.DataArray:
    """Represent optimization bounds as an xarray DataArray."""
    return xr.DataArray(
        bounds.detach().cpu().numpy(),
        dims=("bound", "component"),
        coords={"bound": ["min", "max"], "component": list(component_names)},
        name=variable_name,
    )


def build_single_task_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    standardize: bool = True,
):
    """Construct an *unfitted* SingleTaskGP. Used to rebuild saved models."""
    botorch = import_botorch()
    train_x = torch.as_tensor(train_x)
    train_y = torch.as_tensor(train_y)
    outcome_transform = botorch["Standardize"](m=train_y.shape[-1]) if standardize else None
    model_kwargs: dict[str, Any] = {
        "train_X": train_x,
        "train_Y": train_y,
        "outcome_transform": outcome_transform,
    }

    try:
        return botorch["SingleTaskGP"](**model_kwargs)
    except Exception as exc:
        # BoTorch's validation errors otherwise provide no indication of which
        # observations were supplied to the model.  Log the tensors in their
        # entirety so a queued task failure can be diagnosed from its log.
        if "Input data contains NaN values." in str(exc):
            logger.error(
                "SingleTaskGP rejected inputs with NaN values: "
                "train_x=%s, train_y=%s, standardize=%s",
                train_x,
                train_y,
                standardize,
            )
        raise


def fit_single_task_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    standardize: bool = True,
):
    botorch = import_botorch()
    model = build_single_task_gp(train_x, train_y, standardize=standardize)
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
        def __init__(self) -> None:
            super().__init__()
            self.posterior_mean = botorch["PosteriorMean"](model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.posterior_mean(x)

    candidates, values = botorch["optimize_acqf"](
        acq_function=_TransformedPosteriorMean(),
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


def call_with_supported_kwargs(function, *args, **context):
    """Call ``function(*args, **kw)`` passing only the ``context`` keys its signature accepts.

    A function declaring ``**kwargs`` receives the whole context.
    """
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):  # builtins / C callables: pass positional args only
        return function(*args)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return function(*args, **context)
    positional = {name for name, _ in zip(parameters, args)}
    accepted = {k: v for k, v in context.items() if k in parameters and k not in positional}
    return function(*args, **accepted)


def accepted_context_keys(function, candidates: Sequence[str]) -> set[str]:
    """Subset of ``candidates`` that ``function`` accepts as keyword arguments."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return set()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return set(candidates)
    return {k for k in candidates if k in parameters}


class ScoreFunctionAcquisition(torch.nn.Module):
    """Adapt a user score function to BoTorch's acquisition interface.

    The wrapped function scores one candidate set: it receives ``X`` with shape
    ``(q, d)`` (``q`` jointly proposed design points, no batch dimension) and returns a
    single scalar, higher being better. This module adds BoTorch's batch handling: a
    ``(b, q, d)`` input is scored one candidate set at a time and returned as ``(b,)``.

    ``X`` is a ``torch.Tensor`` so that torch-based functions are optimized with exact
    gradients. Functions that work on NumPy arrays also work: if the function cannot
    handle a tensor that requires gradients, or returns something without autograd
    history, it is called on a detached tensor (which NumPy accepts as an array) and
    :attr:`differentiable` is set to ``False`` so that the optimizer switches to
    finite differences.

    Extra keyword arguments the function declares in its signature are filled from
    ``context`` (runtime objects such as ``model`` and ``best_f``) and ``kwargs``
    (static user settings); everything else in those mappings is ignored.
    """

    def __init__(
        self,
        function,
        context: Mapping[str, Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not callable(function):
            raise TypeError(f"score function must be callable, got {type(function).__name__}")
        self.function = function
        self.context = dict(kwargs or {})
        self.context.update(context or {})
        self.differentiable: bool | None = None

    def _call(self, X: torch.Tensor):
        return call_with_supported_kwargs(self.function, X, **self.context)

    def _scalar(self, result, X: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(result if isinstance(result, torch.Tensor) else np.asarray(result))
        if value.numel() != 1:
            raise ValueError(
                f"score function must return a single scalar per candidate set of shape {tuple(X.shape)}, "
                f"got {value.numel()} values"
            )
        return value.reshape(()).to(dtype=X.dtype, device=X.device)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Score one ``(q, d)`` candidate set, returning a scalar tensor."""
        X = torch.as_tensor(X)
        if X.ndim != 2:
            raise ValueError(f"score expects a (q, d) tensor, got shape {tuple(X.shape)}")
        if self.differentiable is not False and torch.is_grad_enabled() and X.requires_grad:
            try:
                result = self._call(X)
            except (RuntimeError, TypeError):
                result = None
            if isinstance(result, torch.Tensor) and result.grad_fn is not None:
                self.differentiable = True
                return self._scalar(result, X)
            self.differentiable = False
        return self._scalar(self._call(X.detach()), X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 2:  # a single (q, d) candidate set
            return self.score(X)
        batch_shape, q, d = X.shape[:-2], X.shape[-2], X.shape[-1]
        values = [self.score(x) for x in X.reshape(-1, q, d)]
        return torch.stack(values).reshape(batch_shape)


def build_acquisition(acquisition_kind: str, model, best_f: float):
    """Instantiate one of the built-in BoTorch acquisition functions.

    ``acquisition_kind`` is ``"logei"`` (analytic LogEI) or ``"qlogei"`` (Monte Carlo
    qLogEI for joint batches).
    """
    botorch = import_botorch()
    if acquisition_kind == "qlogei":
        return botorch["qLogExpectedImprovement"](model=model, best_f=best_f)
    if acquisition_kind == "logei":
        return botorch["LogExpectedImprovement"](model=model, best_f=best_f)
    raise ValueError(f"Unknown acquisition_kind {acquisition_kind!r}; expected 'logei' or 'qlogei'")


def evaluate_acquisition_on_grid(acq_function, candidate_x: torch.Tensor, q: int = 1) -> torch.Tensor:
    """Evaluate an acquisition function at every row of ``candidate_x`` ``(n, d)``.

    Points are evaluated individually when ``q <= 1``. For joint acquisitions with
    ``q > 1`` the grid is scanned with a sliding window of ``q`` consecutive points;
    windows that do not fit are padded with ``-inf``.
    """
    if q <= 1:
        return acq_function(candidate_x.unsqueeze(-2)).detach()

    values = []
    total = candidate_x.shape[0]
    for start in range(0, total - q + 1):
        batch = candidate_x[start : start + q]
        values.append(acq_function(batch.unsqueeze(0)).reshape(()))

    padded = torch.full((total,), float("-inf"), dtype=candidate_x.dtype)
    if values:
        padded[: len(values)] = torch.stack(values).detach()
    return padded


def optimize_acquisition_function(
    acq_function,
    bounds: torch.Tensor,
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 128,
    inequality_constraints: Sequence[tuple[torch.Tensor, torch.Tensor, float]] | None = None,
    batch_initial_conditions: torch.Tensor | None = None,
    with_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Maximize ``acq_function`` over ``bounds`` with BoTorch's ``optimize_acqf``.

    ``with_grad=False`` makes the L-BFGS-B optimizer use finite-difference gradients,
    which is required for score functions that are not differentiable by autograd.
    """
    botorch = import_botorch()
    options = {} if with_grad else {"with_grad": False}

    candidates, values = botorch["optimize_acqf"](
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,
        inequality_constraints=inequality_constraints,
        batch_initial_conditions=batch_initial_conditions,
    )

    return candidates.detach(), values.detach()
