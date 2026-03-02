import copy
import uuid
import warnings
from types import SimpleNamespace
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar
from typing_extensions import Self

import numpy as np
import xarray as xr

import bumps  # pyright: ignore
import bumps.fitproblem  # pyright: ignore
import bumps.fitters  # pyright: ignore
import bumps.names  # pyright: ignore
import bumps.bounds  # pyright: ignore
import sasmodels  # pyright: ignore
import sasmodels.bumps_model  # pyright: ignore
import sasmodels.core  # pyright: ignore
import sasmodels.data  # pyright: ignore[ruleName]

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify

try:
    from AFL.automation.APIServer.Client import Client
except ImportError:
    warnings.warn('Could not import AFL-automation. Will not be able to run in client-server mode.',stacklevel=2)

@dataclass
class FitParameter:
    """Represents a parameter to be fitted in a SAS model."""
    value: float
    bounds: tuple[float, float] | None = None
    fixed: bool = False
    error: float | None = None


class SASModel:
    """A wrapper class for sasmodels and bumps fitting.
    
    This class provides an interface to the sasmodels library for small-angle scattering
    model fitting using the bumps optimization framework.
    
    Attributes
    ----------
    name : str
        Identifier for this model instance
    sasmodel : str
        Name of the sasmodel to use (e.g., "sphere", "cylinder", "power_law")
    data : sasmodels.data.Data1D
        The experimental data to fit
    parameters : dict[str, FitParameter]
        Dictionary of parameters for the model with their initial values and constraints
    """

    def __init__(
        self,
        name: str,
        data: sasmodels.data.Data1D,
        sasmodel: str,
        parameters: dict[str, dict[str, Any]],
    ) -> None:
        """Initialize a SAS model for fitting.
        
        Parameters
        ----------
        name : str
            Identifier for this model instance
        data : sasmodels.data.Data1D
            The experimental data to fit
        sasmodel : str
            Name of the sasmodel to use (e.g., "sphere", "cylinder", "power_law")
        parameters : dict[str, dict[str, Any]]
            Dictionary of parameters for the model with their initial values and constraints.
            Each parameter should have a "value" key and optionally a "bounds" key.
        """
        self.name = name
        self.data = data
        self.sasmodel: str = sasmodel
        self.kernel = sasmodels.core.load_model(model_name=sasmodel) #pyright: ignore
        self.init_params = copy.deepcopy(x=parameters)
        
        # Results storage
        self.results = None
        self.model_I = None
        self.model_q = None
        self.fit_params = None
        self.model_cov = None

        # Initialize the bumps model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Set up the bumps model with initial parameter values and constraints."""
        self.model = sasmodels.bumps_model.Model(self.kernel)
        
        # Set parameter values and bounds
        for key, param_dict in self.init_params.items():
            self.model.parameters()[key].value = param_dict["value"]
            
            if param_dict.get("bounds") is not None:
                raw_bounds = param_dict["bounds"]
                lower, upper = float(raw_bounds[0]), float(raw_bounds[1])
                if lower > upper:
                    lower, upper = upper, lower

                self.model.parameters()[key].fixed = False
                # Newer bumps expects an iterable (min, max) tuple. Keep a
                # compatibility fallback for environments that still expect
                # a Bounded instance.
                try:
                    self.model.parameters()[key].bounds = (lower, upper)
                except Exception:
                    self.model.parameters()[key].bounds = bumps.bounds.Bounded(lower, upper)
            else:
                self.model.parameters()[key].fixed = True

        # Create the experiment and fit problem
        self.experiment = sasmodels.bumps_model.Experiment(data=self.data, model=self.model)
        self.problem = bumps.fitproblem.FitProblem(self.experiment)

    def copy(self) -> Self:
        """Create a deep copy of this model instance."""
        return copy.deepcopy(self)

    def residuals(self) -> np.ndarray[Any, np.dtype[Any]]:
        """Get the residuals between the model and the data."""
        return self.problem.residuals()

    def __call__(self, params: dict[str, float] | None = None) -> np.ndarray[Any, np.dtype[Any]]:
        """Calculate the model intensity with the current or specified parameters.
        
        Parameters
        ----------
        params : dict[str, float] | None
            Dictionary of parameter names and values to use for the calculation.
            If None, uses the current parameter values.
            
        Returns
        -------
        np.ndarray
            The calculated model intensity
        """
        if params is not None:
            for param_name, param_value in params.items():
                self.model.parameters()[param_name].value = param_value
            self.experiment.update()
            
        return self.experiment.theory()

    def fit(self, fit_method: dict[str, Any] | None = None) -> Any:
        """Fit the model to the data.
        
        Parameters
        ----------
        fit_method : dict[str, Any] | None
            Dictionary of fitting parameters to pass to bumps.fitters.fit.
            If None, uses default Levenberg-Marquardt parameters.
            
        Returns
        -------
        Any
            The fitting results object from bumps
        """
        # Default fitting method if none provided
        if fit_method is None:
            fit_method = {
                "method": "lm",
                "steps": 1000,
                "ftol": 1.5e-6,
                "xtol": 1.5e-6,
                "verbose": False,
            }

        # If there are no free parameters, treat this as a fixed-parameter
        # model evaluation and skip optimizer invocation (which can fail with
        # empty parameter vectors in newer bumps versions).
        if len(self.problem.labels()) == 0:
            self.results = SimpleNamespace(x=np.array([], dtype=float), dx=np.array([], dtype=float))
            self.fit_params = {}
            self.model_I = self()
            self.model_q = self.data.x[self.data.mask == 0]
            self.model_cov = np.zeros((0, 0), dtype=float)
            return self.results
        
        # Try to fit with provided method, fall back to defaults if it fails
        try:
            self.results = bumps.fitters.fit(self.problem, **fit_method)
        except Exception as e:
            print(f"Warning: Fitting failed with provided method: {e}")
            print("Falling back to default Levenberg-Marquardt method")
            self.results = bumps.fitters.fit(
                self.problem,
                method="lm",
                steps=1000,
                ftol=1.5e-6,
                xtol=1.5e-6,
                verbose=True,
            )

        # Store the fitted parameters and model
        self.fit_params = dict(zip(self.problem.labels(), self.problem.getp()))
        self.model_I = self(params=self.fit_params)
        self.model_q = self.data.x[self.data.mask == 0]
        n_params = len(self.problem.labels())

        # Bumps covariance API differs across versions. Prefer direct problem
        # covariance if available, then fit-result covariance, and finally fall
        # back to a diagonal covariance to keep downstream reporting robust.
        cov_matrix = None
        if hasattr(self.problem, "cov"):
            try:
                cov_matrix = self.problem.cov()
            except Exception:
                cov_matrix = None

        if cov_matrix is None and self.results is not None:
            for attr_name in ("cov", "covariance"):
                if hasattr(self.results, attr_name):
                    attr = getattr(self.results, attr_name)
                    try:
                        cov_matrix = attr() if callable(attr) else attr
                    except Exception:
                        cov_matrix = None
                    if cov_matrix is not None:
                        break

        if cov_matrix is None:
            cov_matrix = np.eye(n_params, dtype=float)

        cov_matrix = np.asarray(cov_matrix, dtype=float)
        if cov_matrix.shape != (n_params, n_params):
            cov_matrix = np.eye(n_params, dtype=float)
        self.model_cov = cov_matrix
        
        return self.results

    def get_fit_params(self) -> dict[str, dict[str, Any]]:
        """Get the fitted parameters with their values and uncertainties.
        
        Returns
        -------
        dict[str, dict[str, Any]]
            dict of parameter names and their fitted values and uncertainties
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        params = copy.deepcopy(self.init_params)
        
        # Update with fitted values and errors
        dx = getattr(self.results, "dx", None)
        for idx, param_name in enumerate(self.problem.labels()):
            err_val = np.nan
            if dx is not None:
                try:
                    err_val = float(dx[idx])
                except Exception:
                    err_val = np.nan
            params[param_name] = {
                "value": self.results.x[idx],
                "error": err_val
            }

        # Remove bounds from the output parameters
        for key in list(params):
            if "bounds" in params[key]:
                del params[key]["bounds"]
                
        return params
    
    def get_chisq(self) -> float:
        """Get the chi-squared value of the fit.
        
        Returns
        -------
        float
            The chi-squared value
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        return self.problem.chisq()
    
    def get_fit_summary(self) -> dict[str, Any]:
        """Get a summary of the fit results.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing the fit results summary
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        return {
            "name": self.name,
            "sasmodel": self.sasmodel,
            "chisq": self.get_chisq(),
            "cov": self.model_cov.tolist(),
            "output_fit_params": self.get_fit_params(),
        }


class SASFitter:
    """A class to handle SAS model fitting for multiple datasets and models.
    
    This class manages the fitting of SAS models to experimental data, including
    model selection, parameter optimization, and results analysis.
    
    Attributes
    ----------
    model_inputs : List[dict[str, Any]]
        List of model configurations to fit
    fit_method : dict[str, Any]
        Configuration for the fitting algorithm
    q_min : float
        Minimum q value to include in fitting
    q_max : float
        Maximum q value to include in fitting
    resolution : Any | None
        Resolution function for the data
    """
    
    # Class constants
    DEFAULT_FIT_METHOD: ClassVar[dict[str, Any]] = {
        "method": "lm",
        "steps": 1000,
        "ftol": 1.5e-6,
        "xtol": 1.5e-6,
        "verbose": False,
    }
    
    DEFAULT_MODEL_INPUTS: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "power_law_1",
            "sasmodel": "power_law",
            "q_min": 0.001,
            "q_max": 1.0,
            "fit_params": {
                "power": {"value": 4, "bounds": (3, 4.5)},
                "background": {"value": 1e-4, "bounds": (1e-10, 1e2)},
                "scale": {"value": 1e0, "bounds": (1e-6, 1e4)},
            },
        }
    ]
    
    def __init__(
        self, 
        model_inputs: list[dict[str, Any]] | None = None, 
        fit_method: dict[str, Any] | None = None,
        q_min: float | None = None,
        q_max: float | None = None,
        resolution: Any | None = None,
    ):
        """Initialize the SAS fitter.
        
        Parameters
        ----------
        model_inputs : list[dict[str, Any]] | None
            List of model configurations to fit. If None, uses DEFAULT_MODEL_INPUTS.
        fit_method : dict[str, Any] | None
            Configuration for the fitting algorithm. If None, uses DEFAULT_FIT_METHOD.
        q_min : float
            Minimum q value to include in fitting
        q_max : float
            Maximum q value to include in fitting
        resolution : Any | None
            Resolution function for the data
        """
        self.model_inputs = model_inputs or self.DEFAULT_MODEL_INPUTS
        self.fit_method = fit_method or self.DEFAULT_FIT_METHOD
        self.q_min = q_min or 1e-3
        self.q_max = q_max or 1
        self.resolution = resolution
        
        # Storage for data and results
        self.sasdata: list[sasmodels.data.Data1D] = []
        self.fitted_models: list[list[SASModel]] = []
        self.fit_results: list[list[dict[str, Any]]] = []
        self.report: dict[str, Any] = {}
    
    def set_sasdata(
        self,
        dataset: xr.Dataset,
        sample_dim: str = "sample",
        q_variable: str = "q",
        sas_variable: str = "I",
        sas_err_variable: str = "dI",
        sas_resolution_variable: str | None = None,
    ) -> None:
        """Set the SAS data to be fitted from an xarray Dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            The xarray dataset containing SAS data
        sample_dim : str
            The dimension containing each sample
        q_variable : str
            The variable name for q values
        sas_variable : str
            The variable name for scattering intensity
        sas_err_variable : str
            The variable name for scattering intensity uncertainty
        sas_resolution_variable : str | None
            The variable name for resolution function, if available
        """
        self.sasdata = []
        
        # Extract data for each sample
        for _, sample_data in dataset.groupby(sample_dim, squeeze=False):
            x = sample_data[q_variable].squeeze().values
            y = sample_data[sas_variable].squeeze().values
            dy = sample_data[sas_err_variable].squeeze().values
            
            # Add resolution if available
            dx = None
            if sas_resolution_variable is not None:
                dx = sample_data[sas_resolution_variable].values

            # Create Data1D object and add to list
            self.sasdata.append(sasmodels.data.Data1D(x=x, y=y, dy=dy, dx=dx))
    
    def _create_models(self, data: sasmodels.data.Data1D) -> list[SASModel]:
        """Create SAS models for the given data based on model_inputs.
        
        Parameters
        ----------
        data : sasmodels.data.Data1D
            The SAS data to fit
            
        Returns
        -------
        List[SASModel]
            List of initialized SAS models
        """
        models = []
        
        # Create a model for each model input configuration
        for model_config in self.model_inputs:
            # Get q range for this model
            q_min = model_config.get("q_min", self.q_min)
            q_max = model_config.get("q_max", self.q_max)
            
            # Apply q range mask to data
            data_copy = copy.deepcopy(data)
            data_copy.mask = (data_copy.x < q_min) | (data_copy.x > q_max)
            
            # Create and add the model
            model = SASModel(
                name=model_config["name"],
                data=data_copy,
                sasmodel=model_config["sasmodel"],
                parameters=model_config["fit_params"],
            )
            models.append(model)
        
        return models
    
    def fit_models(
        self, 
        parallel: bool = False, 
        fit_method: dict[str, Any] | None = None
    ) -> tuple[str, xr.Dataset]:
        """Fit all models to all datasets.
        
        Parameters
        ----------
        parallel : bool
            Whether to use parallel processing (not implemented)
        fit_method : dict[str, Any] | None
            Configuration for the fitting algorithm. If None, uses the instance's fit_method.
            
        Returns
        -------
        Tuple[str, xr.Dataset]
            A tuple containing a unique ID for the fit and an xarray Dataset with the results
        """
        if not self.sasdata:
            raise ValueError("No SAS data to fit! Use set_sasdata(...) first.")

        # Use provided fit method or instance default
        fit_method = fit_method or self.fit_method
        
        # Reset results storage
        self.fit_results = []
        self.fitted_models = []
        
        # Fit each dataset
        for data in self.sasdata:
            # Create models for this dataset
            models = self._create_models(data)
            
            # Fit each model
            fitted_models_for_sample = []
            for model in models:
                model.fit(fit_method=fit_method)
                fitted_models_for_sample.append(model.copy())
            
            # Store fitted models and results
            self.fitted_models.append(fitted_models_for_sample)
            self.fit_results.append([model.get_fit_summary() for model in fitted_models_for_sample])
        
        # Build report and results dataset
        report = self._build_report()
        fit_uuid = 'ASAS-' + str(uuid.uuid4())
        output_dataset = self._create_output_dataset(report, fit_uuid)
        
        return fit_uuid, output_dataset
    
    def _build_report(self) -> dict[str, Any]:
        """Build a comprehensive report of the fitting results.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing the fitting report
        """
        report = {
            "fit_method": self.fit_method,
            "model_inputs": self.model_inputs,
            "model_fits": self.fit_results,
        }
        
        # Find best fits for each dataset
        best_fits = {
            "model_name": [],
            "lowest_chisq": [],
            "model_idx": [],
            "model_params": []
        }
        
        for sample_idx, sample_results in enumerate(self.fit_results):
            # Get chi-squared values and model names
            chisq_values = [model["chisq"] for model in sample_results]
            model_names = [model["name"] for model in sample_results]
            
            # Find best model (lowest chi-squared)
            best_idx = int(np.nanargmin(chisq_values))
            
            # Store best model info
            best_fits["model_name"].append(model_names[best_idx])
            best_fits["lowest_chisq"].append(chisq_values[best_idx])
            best_fits["model_idx"].append(best_idx)
            best_fits["model_params"].append(sample_results[best_idx])
        
        # Calculate model probabilities
        probabilities = self._calculate_probabilities()
        
        # Store all chi-squared values
        all_chisq = [[model["chisq"] for model in sample_results] 
                     for sample_results in self.fit_results]
        
        # Add to report
        report["best_fits"] = best_fits
        report["probabilities"] = probabilities
        report["all_chisq"] = all_chisq
        
        self.report = report
        return report
    
    def _calculate_probabilities(self) -> np.ndarray:
        """Calculate the probability of each model for each dataset.
        
        Returns
        -------
        np.ndarray
            Array of model probabilities
        """
        all_probabilities = []
        eps = 1e-300
        
        for sample_results in self.fit_results:
            # Calculate log likelihood for each model
            log_likelihoods = []
            
            for model in sample_results:
                # Get model parameters
                chisq = model["chisq"]
                cov_matrix = np.asarray(model.get("cov", []), dtype=float)

                # Normalize covariance shape across bumps/scipy versions.
                if cov_matrix.ndim == 0:
                    cov_matrix = cov_matrix.reshape(1, 1)
                elif cov_matrix.ndim == 1:
                    if cov_matrix.size == 0:
                        cov_matrix = np.zeros((0, 0), dtype=float)
                    else:
                        cov_matrix = np.diag(cov_matrix)
                elif cov_matrix.ndim > 2:
                    cov_matrix = np.squeeze(cov_matrix)
                    if cov_matrix.ndim != 2:
                        cov_matrix = np.zeros((0, 0), dtype=float)

                if cov_matrix.shape[0] != cov_matrix.shape[1]:
                    # Fall back to smallest square representation.
                    n = min(cov_matrix.shape[0], cov_matrix.shape[1]) if cov_matrix.ndim == 2 else 0
                    cov_matrix = cov_matrix[:n, :n] if n > 0 else np.zeros((0, 0), dtype=float)

                n_params = int(cov_matrix.shape[0])
                
                # Calculate log marginal likelihood
                if n_params == 0:
                    log_det_term = 0.0
                else:
                    try:
                        sign, logdet = np.linalg.slogdet(cov_matrix)
                        if sign <= 0 or not np.isfinite(logdet):
                            log_det_term = np.log(eps)
                        else:
                            log_det_term = float(logdet)
                    except Exception:
                        log_det_term = np.log(eps)

                log_likelihood = (
                    -chisq + 
                    0.5 * log_det_term + 
                    0.5 * n_params * np.log(2 * np.pi)
                )
                
                log_likelihoods.append(log_likelihood)
            
            # Convert to probabilities
            finite = np.asarray(log_likelihoods, dtype=float)
            finite[~np.isfinite(finite)] = -np.inf
            max_ll = np.max(finite)
            if not np.isfinite(max_ll):
                probabilities = np.ones(len(finite), dtype=float) / max(len(finite), 1)
                all_probabilities.append(probabilities)
                continue

            likelihoods = np.exp(finite - max_ll)
            if not np.any(np.isfinite(likelihoods)) or float(np.sum(likelihoods)) <= 0:
                probabilities = np.ones(len(finite), dtype=float) / max(len(finite), 1)
                all_probabilities.append(probabilities)
                continue
            probabilities = likelihoods / np.sum(likelihoods)
            
            all_probabilities.append(probabilities)
        
        return np.array(all_probabilities)
    
    def _create_output_dataset(self, report: dict[str, Any], fit_uuid: str) -> xr.Dataset:
        """Create an xarray Dataset containing all fit results.
        
        Parameters
        ----------
        report : dict[str, Any]
            The fitting report
        fit_uuid : str
            Unique ID for the fit calculation
            
        Returns
        -------
        xr.Dataset
            Dataset containing all fit results
        """
        # Define dimensions
        sample_dim = "sas_fit_sample"
        model_dim = "models"
        
        # Create dataset
        dataset = xr.Dataset()
        
        # Get model names from first sample (assuming all samples have same models)
        model_names = [model["name"] for model in report["model_fits"][0]]
        
        # Add chi-squared values
        dataset["all_chisq"] = xr.DataArray(
            data=report["all_chisq"],
            dims=[sample_dim, model_dim],
            coords={
                sample_dim: np.arange(len(self.sasdata)),
                model_dim: model_names
            }
        )
        dataset["all_chisq"].attrs["fit_calc_id"] = fit_uuid
        
        # Add probabilities
        dataset["probabilities"] = xr.DataArray(
            data=report["probabilities"],
            dims=[sample_dim, model_dim],
            coords={
                sample_dim: np.arange(len(self.sasdata)),
                model_dim: model_names
            }
        )
        
        # Collect parameter values and errors
        param_values = defaultdict(lambda: defaultdict(list))
        param_errors = defaultdict(lambda: defaultdict(list))
        
        # Extract parameter values and errors for each model and sample
        for sample_idx, sample_results in enumerate(report["model_fits"]):
            for model in sample_results:
                model_name = model["name"]
                
                for param_name, param_data in model["output_fit_params"].items():
                    param_key = f"{model_name}_{param_name}"
                    param_values[model_name][param_key].append(param_data["value"])
                    try:
                        param_errors[model_name][param_key].append(param_data["error"])
                    except:
                        param_errors[model_name][param_key].append(0)
        
        # Add parameter values and errors to dataset
        for model_name in param_values:
            # Create parameter dimension
            param_dim = f"{model_name}_params"
            param_names = list(param_values[model_name].keys())
            
            # Add parameter values
            dataset[f"{model_name}_fit_val"] = xr.DataArray(
                data=np.array([param_values[model_name][param] for param in param_names]),
                coords={param_dim: param_names},
                dims=[param_dim, sample_dim]
            )
            dataset[f"{model_name}_fit_val"].attrs["fit_calc_id"] = fit_uuid
            
            # Add parameter errors
            dataset[f"{model_name}_fit_err"] = xr.DataArray(
                data=np.array([param_errors[model_name][param] for param in param_names]),
                coords={param_dim: param_names},
                dims=[param_dim, sample_dim]
            )
            dataset[f"{model_name}_fit_err"].attrs["fit_calc_id"] = fit_uuid
        
        # Add model intensities and residuals
        for model_group in self.fitted_models:
            for model in model_group:
                model_name = model.name
                
                # Add q values if not already added
                q_key = f"fit_q_{model_name}"
                if q_key not in dataset:
                    dataset[q_key] = xr.DataArray(
                        data=model.model_q,
                        dims=[q_key]
                    )
        
        # Collect intensities and residuals
        for sample_idx, model_group in enumerate(self.fitted_models):
            for model in model_group:
                model_name = model.name
                q_key = f"fit_q_{model_name}"
                
                # Add intensity
                if f"fit_I_{model_name}" not in dataset:
                    dataset[f"fit_I_{model_name}"] = xr.DataArray(
                        data=np.zeros((len(self.sasdata), len(model.model_q))),
                        dims=[sample_dim, q_key],
                        coords={
                            sample_dim: np.arange(len(self.sasdata)),
                            q_key: model.model_q
                        }
                    )
                
                # Add residuals
                if f"residuals_{model_name}" not in dataset:
                    dataset[f"residuals_{model_name}"] = xr.DataArray(
                        data=np.zeros((len(self.sasdata), len(model.model_q))),
                        dims=[sample_dim, q_key],
                        coords={
                            sample_dim: np.arange(len(self.sasdata)),
                            q_key: model.model_q
                        }
                    )
                
                # Set values
                dataset[f"fit_I_{model_name}"][sample_idx] = model.model_I
                dataset[f"residuals_{model_name}"][sample_idx] = model.residuals()
        
        return dataset


class AutoSAS(PipelineOp):
    """AutoSAS is a pipeline operation for fitting small-angle scattering (SAS) data.
    
    This class provides an interface to the SAS fitting functionality, allowing users
    to fit various scattering models to experimental data. It can operate either locally
    or by connecting to a remote AutoSAS server.
    
    Attributes
    ----------
    sas_variable : str
        The variable name for scattering intensity in the dataset
    sas_err_variable : str
        The variable name for scattering intensity uncertainty in the dataset
    output_prefix : str
        Prefix to add to output variable names
    q_dim : str
        The dimension name for q values in the dataset
    sas_resolution_variable : float | None
        Resolution function for the data, if available
    sample_dim : str
        The dimension containing each sample
    model_dim : str
        The dimension name for different models
    model_inputs : List[dict[str, Any]] | None
        List of model configurations to fit
    fit_method : dict[str, Any] | None
        Configuration for the fitting algorithm
    server_id : str | None
        Server ID in the format "host:port" for remote execution, or None for local execution
    q_min : float
        Minimum q value to use if not provided in the model_input variable
    q_max : float
        Maximum q value to use if not provided in the model_input variable
    """

    def __init__(
        self,
        sas_variable: str,
        sas_err_variable: str,
        q_variable: str,
        output_prefix: str,
        sas_resolution_variable: Any | None = None,
        sample_dim: str = 'sample',
        model_dim: str = 'models',
        model_inputs: list[dict[str, Any]] | None = None,
        fit_method: dict[str, Any] | None = None,
        server_id: str | None = None,  # Set to None to run locally
        q_min: float | None = None,
        q_max: float | None = None,
        name: str = "AutoSAS",
    ):
        output_variables = ["all_chisq"]
        super().__init__(
            name=name,
            input_variable=[q_variable, sas_variable, sas_err_variable],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        self.server_id = server_id
        self.fit_method = fit_method
        self.model_inputs = model_inputs
        self.results = dict()

        self.q_variable = q_variable
        self.sas_variable = sas_variable
        self.sas_err_variable = sas_err_variable
        self.resolution = sas_resolution_variable

        self.sample_dim = sample_dim
        self.model_dim = model_dim
        self._banned_from_attrs.extend(["AutoSAS_client"])

        self.q_min = q_min
        self.q_max = q_max

    def construct_clients(self):
        """
        Creates a client to talk to the AutoSAS server
        """
        from AFL.automation.APIServer.Client import Client

        if self.server_id is None:
            # No server, will run locally
            return None

        host, port = self.server_id.split(":")
        self.AutoSAS_client = Client(host, port=port)
        self.AutoSAS_client.login("AutoSAS_client")
        self.AutoSAS_client.debug(False)
        return self.AutoSAS_client

    def calculate(self, dataset):
        """
        Run SAS fitting on the input data either locally or via remote server
        """
        # Clone dataset to avoid modifying the original
        sub_dataset = dataset[[self.q_variable, self.sas_variable, self.sas_err_variable]].copy(deep=True)
        
        # Check whether to run locally or remotely
        if self.server_id is None:
            # Run locally using SASFitter
            return self._calculate_local(sub_dataset)
        else:
            # Run remotely using AutoSAS_Driver via API
            return self._calculate_remote(sub_dataset)

    def _calculate_local(self, dataset):
        """
        Run SAS fitting locally using SASFitter class
        """
        # Initialize SASFitter with our configuration
        fitter = SASFitter(
            model_inputs=self.model_inputs,
            fit_method=self.fit_method,
            resolution=self.resolution,
            q_min=self.q_min,
            q_max=self.q_max,
        )
        
        # Set the data to fit
        fitter.set_sasdata(
            dataset=dataset,
            sample_dim=self.sample_dim,
            q_variable=self.q_variable,
            sas_variable=self.sas_variable,
            sas_err_variable=self.sas_err_variable
        )
        
        # Run the fitting
        _, autosas_fit = fitter.fit_models(fit_method=self.fit_method)
        
        # Rename variables and dimensions to match our naming convention
        autosas_fit = autosas_fit.rename_vars({
            'all_chisq': self._prefix_output('all_chisq')
        })
        
        autosas_fit = autosas_fit.rename_dims({
            'sas_fit_sample': self.sample_dim
        }).reset_index('sas_fit_sample').reset_coords()
        
        self.output = autosas_fit
        return self

    def _calculate_remote(self, dataset):
        """
        Run SAS fitting remotely using AutoSAS_Driver via API
        """
        # Create client connection
        self.construct_clients()
        
        # Send dataset to the server
        db_uuid = self.AutoSAS_client.deposit_obj(obj=dataset)

        if self.q_min or self.q_max:
            self.AutoSAS_client.set_config(
                q_min=self.q_min,
                q_max=self.q_max
            )

        # Initialize the input data for fitting
        self.AutoSAS_client.enqueue(
            task_name="set_sasdata",
            db_uuid=db_uuid,
            sample_dim=self.sample_dim,
            q_variable=self.q_variable,
            sas_variable=self.sas_variable,
            sas_err_variable=self.sas_err_variable,
        )

        # Run the fitting
        fit_calc_id = self.AutoSAS_client.enqueue(
            task_name="fit_models", 
            fit_method=self.fit_method,
            interactive=True
        )['return_val']

        # Retrieve the results
        autosas_fit = self.AutoSAS_client.retrieve_obj(uid=fit_calc_id, delete=False)
        
        # Rename variables and dimensions to match our naming convention
        autosas_fit = autosas_fit.rename_vars({
            'all_chisq': self._prefix_output('all_chisq')
        })
        
        autosas_fit = autosas_fit.rename_dims({
            'sas_fit_sample': self.sample_dim
        }).reset_index('sas_fit_sample').reset_coords()
        
        self.output = autosas_fit
        return self

class ModelSelectBestChiSq(PipelineOp):
    """ModelSelectBestChiSq is a pipeline operation for model selection based on the best chi-square value.
    
    This class evaluates competing SAS models based solely on their goodness of fit (chi-square),
    selecting the model with the lowest chi-square value for each sample. This approach prioritizes
    fit quality without considering model complexity.
    
    Note that selecting models based only on chi-square may lead to overfitting, as more complex models
    with more parameters will generally fit better. For model selection that balances fit quality with
    model complexity, consider using ModelSelectAIC or ModelSelectBIC instead.
    
    Attributes
    ----------
    all_chisq_var : str
        The variable name containing chi-square values for all models
    model_names_var : str
        The variable name containing model names
    sample_dim : str
        The dimension containing each sample
    output_prefix : str
        Prefix to add to output variable names
    """
    def __init__(
        self,
        all_chisq_var,
        model_names_var,
        #model_dim,
        sample_dim,
        output_prefix='BestChiSq',
        name="ModelSelection_BestChiSq",
    ):
        
        output_variables = ["labels", "label_names"]
        super().__init__(
            name=name,
            input_variable=[all_chisq_var, model_names_var],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        
        self.sample_dim = sample_dim
        #self.model_dim = model_dim
        self.model_names_var = model_names_var
        
        self.all_chisq_var = all_chisq_var

    def calculate(self, dataset):        
        """Method for selecting the model based on the best chi-squared value"""
        
        self.dataset = dataset.copy(deep=True)
        
        labels = self.dataset[self.all_chisq_var].argmin(self.model_names_var).values
        label_names = np.array([self.dataset[self.model_names_var][i].values for i in labels])
        bestChiSq = self.dataset[self.all_chisq_var].min(self.model_names_var).values

        self.output[self._prefix_output("labels")] = xr.DataArray(
            data=labels,
            dims=[self.sample_dim]
        )
        
        self.output[self._prefix_output("ChiSq")] = xr.DataArray(
            data=bestChiSq,
            dims=[self.sample_dim]
        )

        self.output[self._prefix_output("label_names")] = xr.DataArray(
            data=label_names,
            dims=[self.sample_dim]
        )
        return self


class ModelSelectParsimony(PipelineOp):
    """ModelSelectParsimony is a pipeline operation for selecting SAS models based on parsimony.
    
    This class selects the simplest model that provides an acceptable fit to the data,
    using a chi-squared threshold to determine acceptable fits. It can operate either
    locally or by connecting to a remote AutoSAS server.
    
    The parsimony principle selects the model with the fewest parameters (lowest complexity)
    among those that fit the data adequately. This helps avoid overfitting by preferring
    simpler explanations when they are sufficient.
    
    Attributes
    ----------
    all_chisq_var : str
        The variable name for chi-squared values in the dataset
    model_names_var : str
        The variable name for model names in the dataset
    sample_dim : str
        The dimension containing each sample
    cutoff : float
        The chi-squared threshold for acceptable fits (default: 1.0)
    model_priority : dict[str, int] | None
        Dictionary mapping model names to their complexity (number of parameters)
    model_inputs : list[dict[str, Any]] | None
        List of model configurations, used to determine complexity if model_priority is None
    server_id : str | None
        Server ID in the format "host:port" for remote execution, or None for local execution
    output_prefix : str
        Prefix to add to output variable names
    """
    def __init__(
        self,
        all_chisq_var,
        model_names_var,
        sample_dim,
        cutoff=1.0,
        model_priority=None,
        model_inputs=None,  # Added to support local complexity calculation
        server_id=None,     # Made optional to support local operation
        output_prefix='Parsimony',
        name="ModelSelection_Parsimony",
        **kwargs
    ):
        
        output_variables = ["labels", "label_names"]
        super().__init__(
            name=name,
            input_variable=[all_chisq_var, model_names_var],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        
        self.sample_dim = sample_dim
        self.model_names_var = model_names_var
        self.cutoff = cutoff 
        self.model_priority = model_priority
        self.model_inputs = model_inputs
        self.all_chisq_var = all_chisq_var
        self.server_id = server_id
    
    def construct_clients(self):
        """
        Creates a client to talk to the AutoSAS server
        """
        if self.server_id is None:
            return None
            
        host, port = self.server_id.split(":")
        self.AutoSAS_client = Client(host, port=port)
        self.AutoSAS_client.login("AutoSAS_client")
        self.AutoSAS_client.debug(False)
        return self.AutoSAS_client

    def calculate(self, dataset):        
        """Method for selecting the model based on parsimony given a user defined ChiSq threshold """
        
        self.dataset = dataset.copy(deep=True)
       
        ### default behavior is that complexity is determined by number of free parameters. 
        ### this is an issue if the number of parameters is the same between models. You bank on them having wildly different ChiSq vals
        ### could use a neighbor approach or some more intelligent selection methods
        
            
        # Determine model complexity either from server or local model_inputs
        if self.server_id is not None:
            # Get complexity from server
            self.construct_clients()
            aSAS_config = self.AutoSAS_client.get_config('all', interactive=True)['return_val']
            model_inputs = aSAS_config['model_inputs']
        elif self.model_inputs is not None:
            # Use local model_inputs
            model_inputs = self.model_inputs
        else:
            raise ValueError("Either server_id or model_inputs must be provided to calculate model complexity")
            
        # Calculate complexity based on number of free parameters
        model_params = []
        for model in model_inputs:
            n_params = 0
            for p in model['fit_params']:
                if model['fit_params'][p]['bounds'] != None:
                    n_params += 1
            model_params.append(n_params)
            
        if self.model_priority is None:
            self.model_priority = np.argsort(model_params).tolist()
            
        models = self.dataset[self.model_names_var].values #extract models
        
        # Sort models by priority
        priority_order = [m for _,m in sorted(zip(self.model_priority,models))]
        
        # Sort chi-squared and params accordingly
        sorted_chisq = self.dataset[self.all_chisq_var].sortby(self.model_names_var)
        
        # Find best model per sample based on chi-squared
        best_indices = sorted_chisq.argmin(dim=self.model_names_var)
        best_chisq = sorted_chisq.min(dim=self.model_names_var)
        
        # Iterate over samples to apply parsimony rule
        selected_indices = []
        for i in range(self.dataset.sizes[self.sample_dim]):
            chisq_values = sorted_chisq.isel(sample=i).values
            min_chisq = best_chisq.isel(sample=i).item()

            # Find all models within cutoff
            within_cutoff = np.where(chisq_values - min_chisq <= self.cutoff)[0]

            # Choose the simplest model among them
            simplest_idx = within_cutoff[np.argmin([self.model_priority[i] for i in within_cutoff])]
            
            # print(chisq_values)
            # print(chisq_values - min_chisq)
            # print(within_cutoff)
            # print(self.model_priority)
            # print(simplest_idx,'\n')
            selected_indices.append(simplest_idx)
        
        selected_indices = np.array(selected_indices)

        
        self.output[self._prefix_output("labels")] = xr.DataArray(
            data=selected_indices,
            dims=[self.sample_dim]
        )
        
        self.output[self._prefix_output("label_names")] = xr.DataArray(
            data=[priority_order[i] for i in selected_indices],
            dims=[self.sample_dim]
        )
        return self


class ModelSelectAIC(PipelineOp):
    """ModelSelectAIC is a pipeline operation for model selection using the Akaike Information Criterion (AIC).
    
    This class selects the best model for each sample based on the AIC, which balances
    model fit quality (chi-square) with model complexity (number of parameters). AIC helps
    prevent overfitting by penalizing models with more parameters.
    
    The AIC is calculated as: AIC = chi_square + 2 * k
    where k is the number of free parameters in the model.
    
    The model with the lowest AIC value is selected as the best model for each sample.
    
    Attributes
    ----------
    all_chisq_var : str
        The variable name for chi-square values in the dataset
    model_names_var : str
        The variable name for model names in the dataset
    sample_dim : str
        The dimension containing each sample
    model_inputs : list[dict[str, Any]] | None
        List of model configurations to determine complexity
    server_id : str | None
        Server ID in the format "host:port" for remote execution, or None for local execution
    """
    def __init__(
        self,
        all_chisq_var,
        model_names_var,
        sample_dim,
        model_inputs=None,  # Added to support local complexity calculation
        server_id=None,     # Made optional to support local operation
        output_prefix='AIC',
        name="ModelSelectionAIC",
        **kwargs
    ):
        
        output_variables = ["labels", "label_names", "AIC"]
        super().__init__(
            name=name,
            input_variable=[all_chisq_var, model_names_var],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
       
        self.server_id = server_id
        self.model_inputs = model_inputs
        self.sample_dim = sample_dim
        self.model_names_var = model_names_var
        self.all_chisq_var = all_chisq_var
    
    def construct_clients(self):
        """
        Creates a client to talk to the AutoSAS server
        """
        if self.server_id is None:
            return None
            
        host, port = self.server_id.split(":")
        self.AutoSAS_client = Client(host, port=port)
        self.AutoSAS_client.login("AutoSAS_client")
        self.AutoSAS_client.debug(False)
        return self.AutoSAS_client

    def calculate(self, dataset):        
        """Method for selecting the model based on AIC (Akaike Information Criterion) """
        
        self.dataset = dataset.copy(deep=True)

        # Determine model complexity either from server or local model_inputs
        if self.server_id is not None:
            # Get complexity from server
            self.construct_clients()
            aSAS_config = self.AutoSAS_client.get_config('all', interactive=True)['return_val']
            model_inputs = aSAS_config['model_inputs']
        elif self.model_inputs is not None:
            # Use local model_inputs
            model_inputs = self.model_inputs
        else:
            raise ValueError("Either server_id or model_inputs must be provided to calculate model complexity")
            
        # Calculate number of parameters for each model, d
        d = []
        for model in model_inputs:
            n_params = 0
            for p in model['fit_params']:
                if model['fit_params'][p]['bounds'] != None:
                    n_params += 1
            d.append(n_params)
        d = np.array(d)
        
        ### AIC = 2*d + chisq 
        AIC = np.array([2*i + 2*d for i in self.dataset[self.all_chisq_var].values])

        AIC_labels = np.argmin(AIC, axis=1)
        AIC_label_names = np.array([self.dataset[self.model_names_var][i].values for i in AIC_labels])
        
        self.output['AIC'] = xr.DataArray(
            data=AIC,
            dims=[self.sample_dim, self.model_names_var]
        )

        self.output[self._prefix_output("labels")] = xr.DataArray(
            data=AIC_labels,
            dims=[self.sample_dim]
        )
        
        self.output[self._prefix_output("label_names")] = xr.DataArray(
            data=AIC_label_names,
            dims=[self.sample_dim]
        )

        return self


class ModelSelectBIC(PipelineOp):
    """ModelSelectBIC is a pipeline operation for model selection using the Bayesian Information Criterion (BIC).
    
    This class evaluates competing SAS models based on their goodness of fit (chi-square)
    and model complexity (number of parameters), using the BIC formula:
    BIC = chi-square + k * ln(n), where k is the number of parameters and n is the number of data points.
    
    The BIC penalizes model complexity more strongly than AIC, especially for larger datasets,
    favoring simpler models when the evidence doesn't strongly support additional complexity.
    
    Attributes
    ----------
    all_chisq_var : str
        The variable name containing chi-square values for all models
    model_names_var : str
        The variable name containing model names
    sample_dim : str
        The dimension containing each sample
    model_inputs : list[dict[str, Any]] | None
        List of model configurations to determine parameter counts
    server_id : str | None
        Server ID in the format "host:port" for remote execution, or None for local execution
    output_prefix : str
        Prefix to add to output variable names
    """
    def __init__(
        self,
        all_chisq_var,
        model_names_var,
        sample_dim,
        model_inputs=None,  # Added to support local complexity calculation
        server_id=None,     # Made optional to support local operation
        output_prefix='BIC',
        name="ModelSelectionBIC",
        **kwargs
    ):
        
        output_variables = ["labels", "label_names", "BIC"]
        super().__init__(
            name=name,
            input_variable=[all_chisq_var, model_names_var],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
       
        self.server_id = server_id
        self.model_inputs = model_inputs
        self.sample_dim = sample_dim
        self.model_names_var = model_names_var
        self.all_chisq_var = all_chisq_var
    
    def construct_clients(self):
        """
        Creates a client to talk to the AutoSAS server
        """
        if self.server_id is None:
            return None
            
        host, port = self.server_id.split(":")
        self.AutoSAS_client = Client(host, port=port)
        self.AutoSAS_client.login("AutoSAS_client")
        self.AutoSAS_client.debug(False)
        return self.AutoSAS_client

    def calculate(self, dataset):        
        """Method for selecting the model based on BIC (Bayesian Information Criterion) """
        
        self.dataset = dataset.copy(deep=True)

        bestChiSq_labels = self.dataset[self.all_chisq_var].argmin(self.model_names_var).values
        bestChiSq_label_names = np.array([self.dataset[self.model_names_var][i].values for i in bestChiSq_labels])
        
        # Determine model complexity either from server or local model_inputs
        if self.server_id is not None:
            # Get complexity from server
            self.construct_clients()
            aSAS_config = self.AutoSAS_client.get_config('all', interactive=True)['return_val']
            model_inputs = aSAS_config['model_inputs']
        elif self.model_inputs is not None:
            # Use local model_inputs
            model_inputs = self.model_inputs
        else:
            raise ValueError("Either server_id or model_inputs must be provided to calculate model complexity")
            
        # Calculate number of parameters for each model
        n = []
        for model in model_inputs:
            n_params = 0
            for p in model['fit_params']:
                if model['fit_params'][p]['bounds'] != None:
                    n_params += 1
            n.append(n_params)
        n = np.array(n)
        
        ###  n*ln(len(q))- 2*ln(chisq) = BIC    
        BIC = np.array([n*np.log(len(self.dataset.q.values)) - 2*np.log(i) for i in self.dataset[self.all_chisq_var].values])

        BIC_labels = np.argmin(BIC, axis=1)
        BIC_label_names = np.array([self.dataset[self.model_names_var][i].values for i in BIC_labels])
        
        self.output['BIC'] = xr.DataArray(
            data=BIC,
            dims=[self.sample_dim, self.model_names_var]
        )

        self.output[self._prefix_output("labels")] = xr.DataArray(
            data=BIC_labels,
            dims=[self.sample_dim]
        )
        
        self.output[self._prefix_output("label_names")] = xr.DataArray(
            data=BIC_label_names,
            dims=[self.sample_dim]
        )

        return self

class EstimateSASError(PipelineOp):
    """Estimate the error in measured intensity from a 1D SAS curve"""
    def __init__(self, input_variable, output_variable, method='percent',percent_error=0.05,name='EstimateSASError'):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)
        self.method = method
        self.percent_error = percent_error

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)
        
        if self.method=='percent':
            error = data1.pipe(lambda x: x*self.percent_error)
        elif self.method=='sqrt':
            error = data1.pipe(np.sqrt)
        else:
            raise ValueError('Estimation method must be one of: percent, sqrt')
            
        self.output[self.output_variable] = error
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Calculated SAS error using method '{self.method}'"

        return self



class ModelSelectMostProbable(PipelineOp):
    """ModelSelectMostProbable is a pipeline operation for selecting the most probable model.
    
    This class selects the model with the highest probability for each sample by 
    argmaxing over the 'probabilities' data variable.
    
    Attributes
    ----------
    model_names_var : str
        The variable name for model names in the dataset
    sample_dim : str
        The dimension containing each sample
    model_inputs : list[dict[str, Any]] | None
        List of model configurations (optional)
    server_id : str | None
        Server ID in the format "host:port" for remote execution, or None for local execution
    output_prefix : str
        Prefix to add to output variable names
    """
    
    def __init__(
        self,
        model_names_var,
        sample_dim,
        output_prefix='MostProbable',
        name="ModelSelection_MostProbable",
        **kwargs
    ):
        output_variables = ["labels", "label_names"]
        super().__init__(
            name=name,
            input_variable=[model_names_var],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        
        self.sample_dim = sample_dim
        self.model_names_var = model_names_var


    def calculate(self, dataset):
        """Method for selecting the model with the highest probability for each sample
        
        Raises
        ------
        ValueError
            If 'probabilities' variable is not present in the dataset
        """
        
        self.dataset = dataset.copy(deep=True)
        
        # Check if 'probabilities' variable exists in the dataset
        if 'probabilities' not in self.dataset:
            raise ValueError(
                f"The 'probabilities' variable is required for {self.__class__.__name__}. "
                "Please ensure the dataset contains a 'probabilities' variable with model probabilities."
            )
        
        # Determine the most probable model by argmaxing over probabilities
        probabilities = self.dataset['probabilities']
        
        # Find the index of the maximum probability for each sample
        most_probable_indices = probabilities.argmax(dim=self.model_names_var)
        print(most_probable_indices) 
        # Get the corresponding model names 
        model_names = self.dataset[self.model_names_var]
        most_probable_labels = model_names.isel(**{self.model_names_var: most_probable_indices})
        
        self.output[self._prefix_output("labels")] = xr.DataArray(
            data=most_probable_indices.values,
            dims=[self.sample_dim]
        )
        
        self.output[self._prefix_output("label_names")] = xr.DataArray(
            data=most_probable_labels.values,
            dims=[self.sample_dim]
        )
        return self
