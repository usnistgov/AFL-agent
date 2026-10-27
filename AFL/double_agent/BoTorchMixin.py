"""BoTorch machinery shared by the GP regressors, classifiers and acquisition ops.

Everything BoTorch specific lives on :class:`BoTorchMixin`: the lazy import of the
optional dependency, the tensor helpers that move data between ``xarray`` and
``torch``, GP construction, and the bound helpers that read a
:class:`~AFL.double_agent.PipelineOp.PipelineOp`'s own attributes so an op can call
``self.candidates(dataset)`` instead of threading the same variable and dimension
names through every call. :class:`ScoreFunctionAcquisition` is the one module-level
object: a ``torch.nn.Module`` adapter for user score functions.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import xarray as xr

ObjectiveDirection = Literal["maximize", "minimize"]
#: Optimization bounds for BoTorch ops: ``{component: {"min": lo, "max": hi}}``,
#: ``{component: (lo, hi)}`` or an ordered sequence of ``(lo, hi)`` pairs.
Bounds = Union[Mapping[str, Union[Mapping[str, float], Sequence[float]]], Sequence[Sequence[float]]]
Constraints = Sequence[tuple[torch.Tensor, torch.Tensor, float]]

logger = logging.getLogger(__name__)

__all__ = ["BoTorchMixin", "Bounds", "ObjectiveDirection", "ScoreFunctionAcquisition"]


# --------------------------------------------------------------------------------------
# user score functions as BoTorch acquisitions
# --------------------------------------------------------------------------------------
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

    ``kwargs`` are the static ``args`` of the acquisition spec, with any
    :class:`~AFL.double_agent.variables.Variable` placeholders already resolved; the
    function is called as ``function(X, **kwargs)``.
    """

    def __init__(self, function, kwargs: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if not callable(function):
            raise TypeError(f"score function must be callable, got {type(function).__name__}")
        self.function = function
        self.kwargs = dict(kwargs or {})
        self.differentiable: bool | None = None

    def _call(self, X: torch.Tensor):
        return self.function(X, **self.kwargs)

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


# --------------------------------------------------------------------------------------
# the mixin
# --------------------------------------------------------------------------------------
class BoTorchMixin:
    """All BoTorch machinery an op needs, as methods on one class.

    Mix this into a :class:`~AFL.double_agent.PipelineOp.PipelineOp` subclass (list it
    *before* the op base class) and call :meth:`_init_botorch` at the end of
    ``__init__``.

    The class has two layers. **Stateless helpers** (static/class methods such as
    :meth:`optimization_bounds_to_tensor`, :meth:`fit_single_task_gp`,
    :meth:`evaluate_acquisition_on_grid`) take explicit arguments and can be called
    on the class itself, e.g. ``BoTorchMixin.fit_single_task_gp(train_x, train_y)``.
    **Bound helpers** (:meth:`candidates`, :meth:`bounds_tensor`, :meth:`load_model`,
    :meth:`optimize_acqf`, ...) read the following attributes, which the op sets itself:

    ``grid_variable``, ``grid_dim``
        Candidate grid and the dimension indexing its points; the last remaining
        dimension (or ``component_dim`` if the op defines one) indexes components.
    ``feature_input_variable``, ``predictor_input_variable``, ``sample_dim``, ``objective_direction``
        Only needed by :meth:`training_data` (regressors).
    ``bounds``, ``is_simplex``, ``model_variable``, ``model_uri``, ``num_restarts``, ``raw_samples``
        Set by :meth:`_init_botorch`.

    The mixin does not define ``__init__`` so it composes with any ``PipelineOp``
    constructor signature.
    """

    #: dimension and coordinate names used for the ``bounds`` output
    BOUNDS_OUTPUT = "bounds"

    # -- stateless helpers --------------------------------------------------------------
    @staticmethod
    def import_botorch():
        """Import the BoTorch symbols the ops use, with a single helpful error message."""
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

    @staticmethod
    def _as_float_tensor(data: xr.DataArray, leading_dim: str) -> torch.Tensor:
        values = data.transpose(leading_dim, ...).values
        return torch.as_tensor(np.asarray(values), dtype=torch.double)

    @classmethod
    def dataset_to_botorch_training_data(
        cls,
        dataset: xr.Dataset,
        feature_input_variable: str,
        predictor_input_variable: str,
        sample_dim: str,
        objective_direction: ObjectiveDirection = "minimize",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(train_x, train_y)`` double tensors; ``train_y`` is negated when minimizing."""
        train_x = cls._as_float_tensor(dataset[feature_input_variable], sample_dim)
        train_y = cls._as_float_tensor(dataset[predictor_input_variable], sample_dim)

        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)

        if objective_direction == "minimize":
            train_y = -train_y

        return train_x, train_y

    @classmethod
    def dataset_to_botorch_candidates(cls, dataset: xr.Dataset, grid_variable: str, grid_dim: str) -> torch.Tensor:
        """The candidate grid as an ``(n, d)`` double tensor."""
        return cls._as_float_tensor(dataset[grid_variable], grid_dim)

    @staticmethod
    def posterior_to_xarray(
        posterior,
        grid_index: xr.DataArray,
        output_prefix: str,
        objective_direction: ObjectiveDirection = "maximize",
    ) -> xr.Dataset:
        """``{prefix}_mean`` / ``{prefix}_variance`` over the grid, mean in original objective units."""
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

    @staticmethod
    def get_observed_best_f(train_y: torch.Tensor, objective_direction: ObjectiveDirection = "maximize") -> float:
        """Best observed target in original objective units (``train_y`` is in maximization space)."""
        best_f = float(train_y.max().detach().cpu().item())
        if objective_direction == "minimize":
            return -best_f
        return best_f

    @staticmethod
    def make_simplex_constraints(n_dim: int) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        r"""Return BoTorch-compatible simplex constraints.

        BoTorch's `optimize_acqf` accepts linear inequality constraints of the form

        .. math::
            \sum_i a_i x_i \ge b.

        A simplex equality :math:`\sum_i x_i = 1` is therefore encoded as the pair

        .. math::
            \sum_i x_i \ge 1, \qquad -\sum_i x_i \ge -1,

        which together are exactly equivalent to the equality constraint.
        """
        indices = torch.arange(n_dim, dtype=torch.long)
        coefficients = torch.ones(n_dim, dtype=torch.double)
        return [(indices, coefficients, 1.0), (indices, -coefficients, -1.0)]

    @staticmethod
    def optimization_bounds_to_tensor(bounds: Bounds, component_names: Sequence[Any] | None = None) -> torch.Tensor:
        """Convert explicit constructor bounds into a ``(2, d)`` BoTorch bounds tensor.

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

    @staticmethod
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

    @classmethod
    def build_single_task_gp(cls, train_x: torch.Tensor, train_y: torch.Tensor, standardize: bool = True):
        """Construct an *unfitted* SingleTaskGP. Used to rebuild saved models."""
        botorch = cls.import_botorch()
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
                    "SingleTaskGP rejected inputs with NaN values: train_x=%s, train_y=%s, standardize=%s",
                    train_x,
                    train_y,
                    standardize,
                )
            raise

    @classmethod
    def fit_single_task_gp(cls, train_x: torch.Tensor, train_y: torch.Tensor, standardize: bool = True):
        """Build and fit a SingleTaskGP by maximizing the exact marginal log likelihood."""
        botorch = cls.import_botorch()
        model = cls.build_single_task_gp(train_x, train_y, standardize=standardize)
        mll = botorch["ExactMarginalLogLikelihood"](model.likelihood, model)
        botorch["fit_gpytorch_mll"](mll)
        return model

    @classmethod
    def build_builtin_acquisition(cls, acquisition_kind: str, model, best_f: float):
        """Instantiate one of the built-in BoTorch acquisition functions.

        ``acquisition_kind`` is ``"logei"`` (analytic LogEI) or ``"qlogei"`` (Monte Carlo
        qLogEI for joint batches).
        """
        botorch = cls.import_botorch()
        if acquisition_kind == "qlogei":
            return botorch["qLogExpectedImprovement"](model=model, best_f=best_f)
        if acquisition_kind == "logei":
            return botorch["LogExpectedImprovement"](model=model, best_f=best_f)
        raise ValueError(f"Unknown acquisition_kind {acquisition_kind!r}; expected 'logei' or 'qlogei'")

    @staticmethod
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

    @property
    def botorch(self) -> dict:
        """The BoTorch symbols, imported on first use so ``import AFL`` works without BoTorch."""
        return self.import_botorch()

    def _init_botorch(
        self,
        *,
        bounds: Optional[Bounds] = None,
        is_simplex: bool = False,
        model_variable: Optional[str] = None,
        model_uri: Optional[str] = None,
        num_restarts: int = 10,
        raw_samples: int = 128,
        require_bounds: bool = False,
    ) -> None:
        """Store the shared BoTorch settings and keep ``model`` out of the JSON args."""
        if require_bounds and bounds is None:
            raise ValueError(f"{self.name} requires explicit `bounds`.")
        self.bounds = bounds
        self.is_simplex = is_simplex
        self.model_variable = model_variable
        self.model_uri = model_uri
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.model = None  # model used in the last calculate()
        if "model" not in self._banned_from_attrs:
            self._banned_from_attrs.append("model")

    # -- bound helpers: data ------------------------------------------------------------
    def training_data(self, dataset: xr.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """``(train_x, train_y)`` for this op's feature/predictor variables, in maximization space."""
        return self.dataset_to_botorch_training_data(
            dataset=dataset,
            feature_input_variable=self.feature_input_variable,
            predictor_input_variable=self.predictor_input_variable,
            sample_dim=self.sample_dim,
            objective_direction=self.objective_direction,
        )

    def candidates(self, dataset: xr.Dataset) -> torch.Tensor:
        """The candidate grid ``grid_variable`` as an ``(n, d)`` tensor."""
        return self.dataset_to_botorch_candidates(dataset, grid_variable=self.grid_variable, grid_dim=self.grid_dim)

    def grid_component_dim(self, dataset: xr.Dataset) -> str:
        """The component dimension of the grid (``self.component_dim`` if the op defines one)."""
        explicit = getattr(self, "component_dim", None)
        if isinstance(explicit, str):
            return explicit
        return dataset[self.grid_variable].dims[-1]

    def component_names(self, dataset: xr.Dataset) -> Optional[np.ndarray]:
        """Component coordinate values of the grid, or ``None`` when the grid has no component coords."""
        grid = dataset[self.grid_variable]
        names = grid.coords.get(self.grid_component_dim(dataset))
        return None if names is None else names.values

    # -- bounds / constraints -----------------------------------------------------------
    def bounds_tensor(self, dataset: xr.Dataset, write_output: bool = True) -> torch.Tensor:
        """``self.bounds`` as a ``(2, d)`` tensor aligned with the grid's components.

        When ``write_output`` is true and the grid has component names, the bounds are
        also written to the ``bounds`` output so downstream ops can read them.
        """
        if self.bounds is None:
            raise ValueError(f"{self.name} requires explicit `bounds`.")
        names = self.component_names(dataset)
        tensor = self.optimization_bounds_to_tensor(self.bounds, component_names=names)
        if write_output and names is not None:
            variable_name = self._prefix_output(self.BOUNDS_OUTPUT)
            self.output[variable_name] = self.optimization_bounds_to_xarray(
                tensor, component_names=names, variable_name=variable_name
            )
        return tensor

    def constraints(self, n_dim: int) -> Optional[list[tuple[torch.Tensor, torch.Tensor, float]]]:
        """Simplex inequality constraints when ``is_simplex`` is set, otherwise ``None``."""
        return self.make_simplex_constraints(n_dim) if self.is_simplex else None

    # -- models -------------------------------------------------------------------------
    def load_model(self, dataset: xr.Dataset):
        """Load the model whose ref an upstream op wrote to ``model_variable``."""
        from AFL.double_agent.torch_io import load_model  # torch_io is optional at import time

        if self.model_variable is None:
            raise ValueError(f"{self.name}: no `model_variable` given; an upstream regressor/classifier must provide one.")
        if self.model_variable not in dataset:
            raise KeyError(
                f"{self.name}: model variable {self.model_variable!r} not found in dataset. "
                "Did the upstream regressor run with model_uri set?"
            )
        self.model = load_model(str(dataset[self.model_variable].values[()]))
        return self.model

    def save_model(self, model, args: Optional[dict] = None, builder=None) -> Optional[str]:
        """Save ``model`` to ``model_uri`` and write the ref to ``{prefix}_model``; no-op without ``model_uri``."""
        if self.model_uri is None:
            return None
        from AFL.double_agent.torch_io import save_model

        ref = save_model(model, self.model_uri, args=args, builder=builder, name=self.name)
        self.output[self._prefix_output("model")] = xr.DataArray(ref)
        return ref

    # -- optimization -----------------------------------------------------------------
    def optimize_acqf(
        self,
        acq_function,
        bounds: torch.Tensor,
        q: int = 1,
        *,
        constraints: Optional[Constraints] = None,
        with_grad: bool = True,
        batch_initial_conditions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Maximize ``acq_function`` over ``bounds`` with BoTorch's ``optimize_acqf``.

        Uses ``self.num_restarts`` / ``self.raw_samples``. ``with_grad=False`` makes the
        L-BFGS-B optimizer use finite-difference gradients, which is required for score
        functions that are not differentiable by autograd. Returns detached
        ``(candidates, values)``.
        """
        options = {} if with_grad else {"with_grad": False}
        candidates, values = self.botorch["optimize_acqf"](
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            options=options,
            inequality_constraints=constraints,
            batch_initial_conditions=batch_initial_conditions,
        )
        return candidates.detach(), values.detach()

    @staticmethod
    def best_grid_point(values: torch.Tensor | np.ndarray, candidate_x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Fallback when optimization fails: the grid point with the largest (finite) value.

        Returns ``(x, value)`` with ``x`` shaped ``(1, d)``.
        """
        values_np = np.asarray(values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values).reshape(-1)
        best_index = int(np.nanargmax(values_np))
        return candidate_x[best_index].unsqueeze(0).detach(), float(values_np[best_index])
