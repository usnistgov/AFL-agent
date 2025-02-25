import time
import copy
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple, TypeVar, ClassVar
from typing_extensions import Self

import numpy as np
import xarray as xr
import pandas as pd

import bumps  # type: ignore
import bumps.fitproblem  # type: ignore
import bumps.fitters  # type: ignore
import bumps.names  # type: ignore
import bumps.bounds  # type: ignore
import sasmodels  # type: ignore
import sasmodels.bumps_model  # type: ignore
import sasmodels.core  # type: ignore
import sasmodels.data  # type: ignore

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify

@dataclass
class FitParameter:
    """Represents a parameter to be fitted in a SAS model."""
    value: float
    bounds: Optional[Tuple[float, float]] = None
    fixed: bool = False
    error: Optional[float] = None


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
    parameters : Dict[str, FitParameter]
        Dictionary of parameters for the model with their initial values and constraints
    """

    def __init__(
        self,
        name: str,
        data: sasmodels.data.Data1D,
        sasmodel: str,
        parameters: Dict[str, Dict[str, Any]],
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
        parameters : Dict[str, Dict[str, Any]]
            Dictionary of parameters for the model with their initial values and constraints.
            Each parameter should have a "value" key and optionally a "bounds" key.
        """
        self.name = name
        self.data = data
        self.sasmodel = sasmodel
        self.kernel = sasmodels.core.load_model(sasmodel)
        self.init_params = copy.deepcopy(parameters)
        
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
                self.model.parameters()[key].fixed = False
                self.model.parameters()[key].bounds = bumps.bounds.Bounded(
                    *param_dict["bounds"]
                )
            else:
                self.model.parameters()[key].fixed = True

        # Create the experiment and fit problem
        self.experiment = sasmodels.bumps_model.Experiment(data=self.data, model=self.model)
        self.problem = bumps.fitproblem.FitProblem(self.experiment)

    def copy(self) -> Self:
        """Create a deep copy of this model instance."""
        return copy.deepcopy(self)

    def residuals(self) -> np.ndarray:
        """Get the residuals between the model and the data."""
        return self.problem.residuals()

    def __call__(self, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Calculate the model intensity with the current or specified parameters.
        
        Parameters
        ----------
        params : Optional[Dict[str, float]]
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

    def fit(self, fit_method: Optional[Dict[str, Any]] = None) -> Any:
        """Fit the model to the data.
        
        Parameters
        ----------
        fit_method : Optional[Dict[str, Any]]
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
        self.model_cov = self.problem.cov()
        
        return self.results

    def get_fit_params(self) -> Dict[str, Dict[str, Any]]:
        """Get the fitted parameters with their values and uncertainties.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of parameter names and their fitted values and uncertainties
        """
        if self.results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        params = copy.deepcopy(self.init_params)
        
        # Update with fitted values and errors
        for idx, param_name in enumerate(self.problem.labels()):
            params[param_name] = {
                "value": self.results.x[idx],
                "error": self.results.dx[idx]
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
    
    def get_fit_summary(self) -> Dict[str, Any]:
        """Get a summary of the fit results.
        
        Returns
        -------
        Dict[str, Any]
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
    model_inputs : List[Dict[str, Any]]
        List of model configurations to fit
    fit_method : Dict[str, Any]
        Configuration for the fitting algorithm
    q_min : float
        Minimum q value to include in fitting
    q_max : float
        Maximum q value to include in fitting
    resolution : Optional[Any]
        Resolution function for the data
    """
    
    # Class constants
    DEFAULT_FIT_METHOD: ClassVar[Dict[str, Any]] = {
        "method": "lm",
        "steps": 1000,
        "ftol": 1.5e-6,
        "xtol": 1.5e-6,
        "verbose": False,
    }
    
    DEFAULT_MODEL_INPUTS: ClassVar[List[Dict[str, Any]]] = [
        {
            "name": "power_law_1",
            "sasmodel": "power_law",
            "q_min": 0.01,
            "q_max": 0.4,
            "fit_params": {
                "power": {"value": 4, "bounds": (3, 4.5)},
                "background": {"value": 1e-4, "bounds": (1e-10, 1e2)},
                "scale": {"value": 1e0, "bounds": (1e-6, 1e4)},
            },
        }
    ]
    
    def __init__(
        self, 
        model_inputs: Optional[List[Dict[str, Any]]] = None, 
        fit_method: Optional[Dict[str, Any]] = None,
        q_min: float = 1e-2,
        q_max: float = 1e-1,
        resolution: Optional[Any] = None,
    ):
        """Initialize the SAS fitter.
        
        Parameters
        ----------
        model_inputs : Optional[List[Dict[str, Any]]]
            List of model configurations to fit. If None, uses DEFAULT_MODEL_INPUTS.
        fit_method : Optional[Dict[str, Any]]
            Configuration for the fitting algorithm. If None, uses DEFAULT_FIT_METHOD.
        q_min : float
            Minimum q value to include in fitting
        q_max : float
            Maximum q value to include in fitting
        resolution : Optional[Any]
            Resolution function for the data
        """
        self.model_inputs = model_inputs or self.DEFAULT_MODEL_INPUTS
        self.fit_method = fit_method or self.DEFAULT_FIT_METHOD
        self.q_min = q_min
        self.q_max = q_max
        self.resolution = resolution
        
        # Storage for data and results
        self.sasdata: List[sasmodels.data.Data1D] = []
        self.fitted_models: List[List[SASModel]] = []
        self.fit_results: List[List[Dict[str, Any]]] = []
        self.report: Dict[str, Any] = {}
    
    def set_sasdata(
        self,
        dataset: xr.Dataset,
        sample_dim: str = "sample",
        q_variable: str = "q",
        sas_variable: str = "I",
        sas_err_variable: str = "dI",
        sas_resolution_variable: Optional[str] = None,
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
        sas_resolution_variable : Optional[str]
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
    
    def _create_models(self, data: sasmodels.data.Data1D) -> List[SASModel]:
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
        fit_method: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, xr.Dataset]:
        """Fit all models to all datasets.
        
        Parameters
        ----------
        parallel : bool
            Whether to use parallel processing (not implemented)
        fit_method : Optional[Dict[str, Any]]
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
        fit_uuid = 'AS-' + str(uuid.uuid4())
        output_dataset = self._create_output_dataset(report, fit_uuid)
        
        return fit_uuid, output_dataset
    
    def _build_report(self) -> Dict[str, Any]:
        """Build a comprehensive report of the fitting results.
        
        Returns
        -------
        Dict[str, Any]
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
        
        for sample_results in self.fit_results:
            # Calculate log likelihood for each model
            log_likelihoods = []
            
            for model in sample_results:
                # Get model parameters
                chisq = model["chisq"]
                cov_matrix = np.array(model["cov"])
                n_params = len(cov_matrix)
                
                # Calculate log marginal likelihood
                log_likelihood = (
                    -chisq + 
                    0.5 * np.log(np.linalg.det(cov_matrix)) + 
                    0.5 * n_params * np.log(2 * np.pi)
                )
                
                log_likelihoods.append(log_likelihood)
            
            # Convert to probabilities
            likelihoods = np.exp(log_likelihoods)
            probabilities = likelihoods / np.sum(likelihoods)
            
            all_probabilities.append(probabilities)
        
        return np.array(all_probabilities)
    
    def _create_output_dataset(self, report: Dict[str, Any], fit_uuid: str) -> xr.Dataset:
        """Create an xarray Dataset containing all fit results.
        
        Parameters
        ----------
        report : Dict[str, Any]
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
                    param_errors[model_name][param_key].append(param_data["error"])
        
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
    def __init__(
        self,
        sas_variable,
        sas_err_variable,
        resolution,
        output_prefix,
        q_dim,
        sample_dim,
        model_dim,
        model_inputs=None,
        fit_method=None,
        server_id=None,  # Set to None to run locally
        name="AutoSAS_fit",
    ):
        output_variables = ["all_chisq"]
        super().__init__(
            name=name,
            input_variable=[q_dim, sas_variable, sas_err_variable],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        self.server_id = server_id
        self.fit_method = fit_method
        self.model_inputs = model_inputs
        self.results = dict()

        self.q_dim = q_dim
        self.sas_variable = sas_variable
        self.sas_err_variable = sas_err_variable
        self.resolution = resolution

        self.sample_dim = sample_dim
        self.model_dim = model_dim
        self._banned_from_attrs.extend(["AutoSAS_client"])

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
        sub_dataset = dataset[[self.q_dim, self.sas_variable, self.sas_err_variable]].copy(deep=True)
        
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
            resolution=self.resolution
        )
        
        # Set the data to fit
        fitter.set_sasdata(
            dataset=dataset,
            sample_dim=self.sample_dim,
            q_variable=self.q_dim,
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

        # Initialize the input data for fitting
        self.AutoSAS_client.enqueue(
            task_name="set_sasdata",
            db_uuid=db_uuid,
            sample_dim=self.sample_dim,
            q_variable=self.q_dim,
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
    def __init__(
        self,
        all_chisq_var,
        model_names_var,
        sample_dim,
        cutoff_threshold=1.0,
        model_complexity=None,
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
        self.cutoff_threshold = cutoff_threshold 
        self.model_complexity = model_complexity
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

        bestChiSq_labels = self.dataset[self.all_chisq_var].argmin(self.model_names_var)
        bestChiSq_label_names = np.array([self.dataset[self.model_names_var][i].values for i in bestChiSq_labels.values])
        
        ### default behavior is that complexity is determined by number of free parameters. 
        ### this is an issue if the number of parameters is the same between models. You bank on them having wildly different ChiSq vals
        ### could use a neighbor approach or some more intelligent selection methods
        if self.model_complexity is None:
            print('aggregating complexity')
            
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
            order = []
            for model in model_inputs:
                n_params = 0
                for p in model['fit_params']:
                    if model['fit_params'][p]['bounds'] != None:
                        n_params += 1
                order.append(n_params)
            print(order)
            print(np.argsort(order))
            self.model_complexity = np.argsort(order).tolist()

        # As written in dev full of jank...
        replacement_labels = bestChiSq_labels.copy(deep=True)
        all_chisq = self.dataset[self.all_chisq_var]
        sorted_chisq = all_chisq.sortby(self.model_names_var, ascending=False).values

        min_diff_chisq = np.array([row[1] - row[0] for row in sorted_chisq])
        next_best_idx = np.array([np.argpartition(row,1)[1] for row in all_chisq])

        for idx in range(len(replacement_labels)):
            chisq_set = all_chisq.min(dim=self.model_names_var).values

            if (min_diff_chisq[idx] <= self.cutoff_threshold):
                best_model_index = replacement_labels[idx]
                next_best_index = next_best_idx[idx]
                bm_rank = self.model_complexity.index(best_model_index)
                nbm_rank = self.model_complexity.index(next_best_index)
                
                if (bm_rank > nbm_rank):
                    replacement_labels[idx] = next_best_index

        self.output[self._prefix_output("labels")] = xr.DataArray(
            data=replacement_labels,
            dims=[self.sample_dim]
        )
        
        self.output[self._prefix_output("label_names")] = xr.DataArray(
            data=[self.dataset[self.model_names_var].values[i] for i in replacement_labels],
            dims=[self.sample_dim]
        )
        return self


class ModelSelectAIC(PipelineOp):
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
        
        ### chisq + 2*ln(d) = AIC    
        AIC = np.array([2*np.log(i) + 2*n for i in self.dataset[self.all_chisq_var].values])

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


class ModelSelectBayesianModelComparison(PipelineOp):
    """Uses a Bayesian model comparison approach to calculating probabilities given a set of models and outputs"""
    def __init__(
        self,
    ):
        return
    
    def calculate(self):
        return 

