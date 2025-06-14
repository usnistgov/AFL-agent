import ipywidgets
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import copy
import uuid
import json
import sasmodels.data
import sasmodels.direct_model
import sasmodels.core
import ipywidgets as widgets
from IPython.display import display, JSON
import re
import xarray as xr

# Make sure plotly works well in Jupyter
#from plotly.offline import init_notebook_mode
#init_notebook_mode(connected=True)

class AutoSASWidget:
    """Interactive widget for building SAS model inputs for AutoSAS.
    
    This widget allows users to create, configure and visualize SAS models
    through an interactive interface. Users can add multiple models as tabs,
    adjust their parameters, and see real-time visualization of the models.
    
    The interface follows an MVC pattern with separate model and view classes.
    """
    
    def __init__(
        self,
        q_range: Optional[np.ndarray] = None,
        default_models: Optional[List[Dict[str, Any]]] = None,
        data: Optional[xr.Dataset] = None,
        model_inputs: Optional[List[Dict[str, Any]]] = None,
        custom_models = None,
    ):
        """Initialize the AutoSASWidget.
        
        Parameters
        ----------
        q_range : Optional[np.ndarray]
            Array of q values to use for model visualization.
            If None, a default range will be created.
        default_models : Optional[List[Dict[str, Any]]]
            List of model configurations to initialize with.
        data : Optional[xr.Dataset]
            Dataset containing experimental scattering data.
        model_inputs : Optional[List[Dict[str, Any]]]
            List of model configurations to initialize with. Takes precedence over default_models.
        """
        # Set up default q_range if not provided
        if q_range is None:
            self.q_range = np.logspace(-3, 0, 100)
        else:
            self.q_range = q_range
            
        # Create the model and view
        self.model = AutoSASWidget_Model(q_range=self.q_range, default_models=default_models,custom_models=custom_models)
        self.view = AutoSASWidget_View(available_models=self.model.available_sasmodels, data=data)
        
        # Connect model and view
        self._setup_callbacks()
        
        # Load model inputs if provided
        if model_inputs:
            self.load_model_inputs(model_inputs)
    
    def _setup_callbacks(self):
        """Connect UI events to model methods."""
        # Add model tab button callback
        self.view.button["add_model"].on_click(self.add_model_callback)
        
        # Remove model tab button callback
        self.view.button["remove_model"].on_click(self.remove_model_callback)
        
        # Export button callback
        self.view.button["export"].on_click(self.export_callback)
    
    def add_model_callback(self, b):
        """Handle add model button click."""
        # Get the model name from dropdown
        model_name = self.view.dropdown["model_type"].value
        
        # Add model to the model component
        model_id = self.model.add_model(model_name)
        
        # Add a new tab to the view
        self.view.add_model_tab(model_id, model_name)
        
        # Set up parameter controls for the new model
        self._setup_parameter_controls(model_id)
        
        # Update the plot
        self.update_plot(model_id)

        self.view.figures[model_id].update_layout(height=549)
        self.view.figures[model_id].update_layout(height=550)

        # Set the new tab as active
        self.view.tabs.selected_index = len(self.view.tab_ids) - 1
    
    def remove_model_callback(self, b):
        """Handle remove model button click."""
        # Get currently selected tab
        current_tab = self.view.tabs.selected_index
        if current_tab >= 0:
            # Get model ID
            model_id = self.view.tab_ids[current_tab]
            
            # Remove from model
            self.model.remove_model(model_id)
            
            # Remove tab from view
            self.view.remove_model_tab(current_tab)
            
            # Update titles for remaining tabs
            for i, tab_id in enumerate(self.view.tab_ids):
                # Get the base model name (remove the _N suffix)
                base_name = self.model.models[tab_id]["sasmodel"]
                # Update the tab title with new index
                self.view.tabs.set_title(i, f"{base_name}_{i + 1}")
                # Update the model name in the model data
                self.model.models[tab_id]["name"] = f"{base_name}_{i + 1}"
    
    def _setup_parameter_controls(self, model_id):
        """Set up parameter controls for a model."""
        # Get all parameters for this model
        params = self.model.get_model_parameters(model_id)
        
        # Set up q_min and q_max controls
        q_min_control = self.view.parameter_controls[model_id]["q_min"]
        q_max_control = self.view.parameter_controls[model_id]["q_max"]
        
        # Get current q range from model
        model_data = self.model.models[model_id]
        q_min_control.value = model_data["q_min"]
        q_max_control.value = model_data["q_max"]
        
        # Set up callbacks for q range controls
        q_min_control.observe(
            lambda change, mid=model_id: 
                self.q_range_change_callback(change, mid, "q_min"),
            names="value"
        )
        
        q_max_control.observe(
            lambda change, mid=model_id: 
                self.q_range_change_callback(change, mid, "q_max"),
            names="value"
        )
        
        # Add controls for each parameter
        for param_name, param_info in params.items():
            # Determine if this parameter should be included in AutoSAS
            use_autosas = param_info.get("autosas", False)
            
            # Determine if this parameter should use bounds
            use_bounds = param_info.get("use_bounds", False)
            
            # Add the control
            self.view.add_parameter_control(
                model_id, 
                param_name, 
                param_info["value"],
                param_info.get("bounds"),
                use_autosas=use_autosas,
                use_bounds=use_bounds
            )
            
            # Get the control and set up callback
            control = self.view.get_parameter_control(model_id, param_name)
            control.observe(
                lambda change, mid=model_id, pname=param_name: 
                    self.parameter_change_callback(change, mid, pname),
                names="value"
            )
            
            # Get the AutoSAS checkbox and set up callback
            autosas_checkbox = self.view.get_parameter_checkbox(model_id, param_name)
            if autosas_checkbox:
                autosas_checkbox.observe(
                    lambda change, mid=model_id, pname=param_name: 
                        self.parameter_checkbox_callback(change, mid, pname),
                    names="value"
                )
            
            # Get the bounds checkbox and set up callback
            bounds_checkbox = self.view.get_bounds_checkbox(model_id, param_name)
            if bounds_checkbox:
                bounds_checkbox.observe(
                    lambda change, mid=model_id, pname=param_name: 
                        self.parameter_checkbox_callback(change, mid, pname),
                    names="value"
                )
    
    def parameter_change_callback(self, change, model_id, param_name):
        """Handle parameter value change."""
        # Update the model parameter
        self.model.update_parameter(model_id, param_name, change["new"])
        
        # Update the plot
        self.update_plot(model_id)
    
    def parameter_checkbox_callback(self, change, model_id, param_name):
        """Handle parameter checkbox change (fixed/variable)."""
        # Determine which checkbox was changed (AutoSAS or Bounds)
        checkbox_type = change.get('owner').description.lower()
        
        if checkbox_type == 'autosas':
            # Update model parameter to reflect AutoSAS mode (include in model_inputs)
            self.model.models[model_id]["params"][param_name]["autosas"] = change["new"]
        elif checkbox_type == 'bounds':
            # Update model parameter to reflect Bounds mode (show slider and include bounds)
            self.model.models[model_id]["params"][param_name]["use_bounds"] = change["new"]
            
            # Get current controls
            control = self.view.get_parameter_control(model_id, param_name)
            
            # Get parameter bounds
            bounds = self.model.get_parameter_bounds(model_id, param_name)
            
            # If bounds checkbox is checked, replace with slider
            if change["new"]:
                self.view.replace_with_slider(model_id, param_name, control.value, bounds)
            # If bounds checkbox is unchecked, replace with text input
            else:
                self.view.replace_with_text(model_id, param_name, control.value)
        
        # Update the plot after any checkbox change
        self.update_plot(model_id)
    
    def update_plot(self, model_id):
        """Update the plot for a model."""
        # Calculate model intensity
        q, intensity = self.model.calculate_model_intensity(model_id)
        
        # Update the plot
        self.view.update_plot(model_id, q, intensity)
    
    def get_model_inputs(self) -> List[Dict[str, Any]]:
        """Get the model inputs in the format required by AutoSAS.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of model configurations
        """
        result = []
        
        for model_id, model in self.model.models.items():
            # Convert our internal format to AutoSAS format
            fit_params = {}
            # Group parameters by their base name (for polydispersity)
            param_groups = {}
            
            for name, param_info in model["params"].items():
                # Only include parameter if AutoSAS is enabled
                if param_info.get("autosas", False):
                    # Check if this is a polydispersity parameter
                    if name.endswith(('_pd', '_pd_type', '_pd_n', '_pd_nsigma')):
                        # Get the base parameter name
                        base_name = name.rsplit('_pd', 1)[0]
                        if base_name not in param_groups:
                            param_groups[base_name] = {}
                        
                        # Store the polydispersity parameter
                        if name.endswith('_pd'):
                            param_groups[base_name]['pd'] = param_info["value"]
                        elif name.endswith('_pd_type'):
                            param_groups[base_name]['pd_type'] = param_info["value"]
                        elif name.endswith('_pd_n'):
                            param_groups[base_name]['pd_n'] = param_info["value"]
                        elif name.endswith('_pd_nsigma'):
                            param_groups[base_name]['pd_nsigma'] = param_info["value"]
                    else:
                        # Regular parameter
                        param_dict = {"value": param_info["value"]}
                        
                        # Only include bounds if bounds mode is enabled and bounds exist
                        if (param_info.get("use_bounds", False) and 
                            param_info.get("bounds") is not None):
                            param_dict["bounds"] = param_info["bounds"]
                        
                        fit_params[name] = param_dict
            
            # Add polydispersity parameters to their base parameters
            for base_name, pd_params in param_groups.items():
                if base_name in fit_params:
                    # Only add polydispersity if the base parameter is included
                    for pd_key, pd_value in pd_params.items():
                        fit_params[base_name][pd_key] = pd_value
            
            # Create the model config
            model_config = {
                "name": model["name"],
                "sasmodel": model["sasmodel"],
                "q_min": model["q_min"],
                "q_max": model["q_max"],
                "fit_params": fit_params
            }
            
            result.append(model_config)
        
        return result
    
    def run(self):
        """Display the widget interface."""
        display(self.view.main_container)
    
    def export_callback(self, b):
        """Handle export button click."""
        # Get model inputs and store in widget attribute
        self.model_inputs = self.get_model_inputs()
        
        # Show notification
        self.view.notification.value = 'model inputs written to widget.model_inputs'
        self.view.notification.layout.display = 'block'
        
        # Schedule notification to disappear after 5 seconds
        import asyncio
        from IPython.display import display
        
        async def hide_notification():
            await asyncio.sleep(5)
            self.view.notification.layout.display = 'none'
        
        asyncio.create_task(hide_notification())
    
    def load_model_inputs(self, model_inputs: List[Dict[str, Any]]) -> None:
        """Load model inputs from an existing configuration.
        
        This allows users to load existing AutoSAS model configurations
        into the widget for further editing and visualization.
        
        Parameters
        ----------
        model_inputs : List[Dict[str, Any]]
            List of model configurations in AutoSAS format
        """
        # Clear existing models first
        # Get all tab IDs
        tab_ids = self.view.tab_ids.copy()
        
        # Remove all tabs
        for model_id in tab_ids:
            self.model.remove_model(model_id)
            
        # Clear view tabs
        self.view.clear_tabs()
        
        # Add models from configuration
        for model_config in model_inputs:
            # Add to model
            model_id = self.model.add_model_from_config(model_config)
            
            # Add to view
            self.view.add_model_tab(model_id, model_config["name"])
            
            # Set up parameter controls
            self._setup_parameter_controls(model_id)
            
            # Update plot
            self.update_plot(model_id)
    
    def q_range_change_callback(self, change, model_id, range_type):
        """Handle q range change."""
        # Update the model q range
        if range_type == "q_min":
            self.model.update_q_range(model_id, q_min=change["new"])
        else:
            self.model.update_q_range(model_id, q_max=change["new"])
        
        # Update the plot
        self.update_plot(model_id)


class AutoSASWidget_Model:
    """Model component for AutoSASWidget.
    
    Handles the data and business logic for the SAS models.
    """
    
    def __init__(
        self,
        q_range: np.ndarray,
        default_models: Optional[List[Dict[str, Any]]] = None,
        custom_models = None
    ):
        """Initialize the model component.
        
        Parameters
        ----------
        q_range : np.ndarray
            Array of q values to use for model visualization
        default_models : Optional[List[Dict[str, Any]]]
            List of model configurations to initialize with
        """
        self.q_range = q_range
        self.models = {}

        self.custom_models = custom_models or []
        self.available_sasmodels = custom_models + self._get_available_sasmodels() 
        
        # Initialize with default models if provided
        if default_models:
            for model_config in default_models:
                self.add_model_from_config(model_config)
    
    def _get_available_sasmodels(self) -> List[str]:
        """Get list of available SAS models from sasmodels.
        
        Returns
        -------
        List[str]
            List of available model names
        """
        try:
            return sorted(sasmodels.core.list_models())
        except Exception as e:
            print(f"Error getting model list: {e}")
            # Fallback to a known list of common models
            return [
                "sphere", "cylinder", "ellipsoid", "core_shell_sphere",
                "core_shell_cylinder", "power_law", "fractal", "pearl_necklace",
                "guinier", "guinier_porod", "unified_power_rg", "dab", 
                "broad_peak", "two_lorentzian", "teubner_strey"
            ]
    
    def add_model(self, model_name: str) -> str:
        """Add a new SAS model.
        
        Parameters
        ----------
        model_name : str
            Name of the SAS model
            
        Returns
        -------
        str
            Unique ID for the created model
        """
        # Generate a unique ID for this model
        model_id = str(uuid.uuid4())
        
        # Create model kernel
        kernel = sasmodels.core.load_model(model_name)
        
        # Get polydispersity parameters
        pd_params = kernel.info.parameters.pd_1d if hasattr(kernel.info.parameters, 'pd_1d') else []
        
        # Get default parameters
        params = {}
        for name, value in kernel.info.parameters.defaults.items():
            # Determine reasonable bounds based on the parameter
            # This is a simple heuristic and might need adjustment
            if value > 0:
                bounds = (value / 10, value * 10)
            elif value < 0:
                bounds = (value * 10, value / 10)
            else:
                bounds = (-1, 1)
            
            params[name] = {
                "value": value,
                "bounds": bounds,
                "fixed": False
            }
            
            # Add polydispersity parameters if this parameter supports it
            if name in pd_params:
                # Add width parameter
                params[f"{name}_pd"] = {
                    "value": 0.0,
                    "bounds": (0.0, 1.0),
                    "fixed": False
                }
                
                # Add distribution type parameter
                params[f"{name}_pd_type"] = {
                    "value": "gaussian",
                    "options": ["gaussian", "rectangular", "schulz"],
                    "fixed": False
                }
                
                # Add number of points parameter
                params[f"{name}_pd_n"] = {
                    "value": 35,
                    "bounds": (3, 200),
                    "fixed": False
                }
                
                # Add number of sigmas parameter
                params[f"{name}_pd_nsigma"] = {
                    "value": 3.0,
                    "bounds": (1.0, 10.0),
                    "fixed": False
                }
        
        # Store the model
        self.models[model_id] = {
            "name": f"{model_name}_{len(self.models) + 1}",
            "sasmodel": model_name,
            "kernel": kernel,
            "params": params,
            "q_min": min(self.q_range),
            "q_max": max(self.q_range)
        }
        
        return model_id
    
    def add_model_from_config(self, model_config: Dict[str, Any]) -> str:
        """Add a model from an existing configuration.
        
        Parameters
        ----------
        model_config : Dict[str, Any]
            Model configuration dictionary
            
        Returns
        -------
        str
            Unique ID for the created model
        """
        # Generate a unique ID for this model
        model_id = str(uuid.uuid4())
        
        # Create model kernel
        kernel = sasmodels.core.load_model(model_config["sasmodel"])
        
        # Get polydispersity parameters
        pd_params = kernel.info.parameters.pd_1d if hasattr(kernel.info.parameters, 'pd_1d') else []
        
        # Start with all default parameters
        params = {}
        for name, value in kernel.info.parameters.defaults.items():
            # Determine reasonable bounds based on the parameter
            # This is a simple heuristic and might need adjustment
            if value > 0:
                bounds = (value / 10, value * 10)
            elif value < 0:
                bounds = (value * 10, value / 10)
            else:
                bounds = (-1, 1)
            
            params[name] = {
                "value": value,
                "bounds": bounds,
                "fixed": True,
                "autosas": False,
                "use_bounds": False
            }
            
            # Add polydispersity parameters if this parameter supports it
            if name in pd_params:
                # Add width parameter
                params[f"{name}_pd"] = {
                    "value": 0.0,
                    "bounds": (0.0, 1.0),
                    "fixed": True,
                    "autosas": False,
                    "use_bounds": False
                }
                
                # Add distribution type parameter
                params[f"{name}_pd_type"] = {
                    "value": "gaussian",
                    "options": ["gaussian", "rectangular", "schulz"],
                    "fixed": True,
                    "autosas": False,
                    "use_bounds": False
                }
                
                # Add number of points parameter
                params[f"{name}_pd_n"] = {
                    "value": 35,
                    "bounds": (3, 200),
                    "fixed": True,
                    "autosas": False,
                    "use_bounds": False
                }
                
                # Add number of sigmas parameter
                params[f"{name}_pd_nsigma"] = {
                    "value": 3.0,
                    "bounds": (1.0, 10.0),
                    "fixed": True,
                    "autosas": False,
                    "use_bounds": False
                }
        
        # Now update with the provided parameters from the configuration
        for name, param_info in model_config["fit_params"].items():
            # Update base parameter
            if name in params:
                params[name].update({
                    "value": param_info["value"],
                    "bounds": param_info.get("bounds", params[name]["bounds"]),
                    "fixed": "bounds" not in param_info,
                    "autosas": True,  # If it's in fit_params, it should be included in AutoSAS
                    "use_bounds": "bounds" in param_info
                })
            
            # Handle polydispersity parameters if present
            if any(pd_key in param_info for pd_key in ['pd', 'pd_type', 'pd_n', 'pd_nsigma']):
                if 'pd' in param_info and f"{name}_pd" in params:
                    params[f"{name}_pd"].update({
                        "value": param_info["pd"],
                        "autosas": True
                    })
                if 'pd_type' in param_info and f"{name}_pd_type" in params:
                    params[f"{name}_pd_type"].update({
                        "value": param_info["pd_type"],
                        "autosas": True
                    })
                if 'pd_n' in param_info and f"{name}_pd_n" in params:
                    params[f"{name}_pd_n"].update({
                        "value": param_info["pd_n"],
                        "autosas": True
                    })
                if 'pd_nsigma' in param_info and f"{name}_pd_nsigma" in params:
                    params[f"{name}_pd_nsigma"].update({
                        "value": param_info["pd_nsigma"],
                        "autosas": True
                    })
        
        # Store the model
        self.models[model_id] = {
            "name": model_config["name"],
            "sasmodel": model_config["sasmodel"],
            "kernel": kernel,
            "params": params,
            "q_min": model_config.get("q_min", min(self.q_range)),
            "q_max": model_config.get("q_max", max(self.q_range))
        }
        
        return model_id
    
    def remove_model(self, model_id: str) -> None:
        """Remove a model.
        
        Parameters
        ----------
        model_id : str
            ID of the model to remove
        """
        if model_id in self.models:
            del self.models[model_id]
    
    def get_model_parameters(self, model_id: str) -> Dict[str, Dict[str, Any]]:
        """Get parameters for a model.
        
        Parameters
        ----------
        model_id : str
            ID of the model
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of parameter information
        """
        if model_id in self.models:
            return self.models[model_id]["params"]
        return {}
    
    def get_parameter_bounds(self, model_id: str, param_name: str) -> Tuple[float, float]:
        """Get bounds for a parameter.
        
        Parameters
        ----------
        model_id : str
            ID of the model
        param_name : str
            Name of the parameter
            
        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds for the parameter
        """
        if model_id in self.models and param_name in self.models[model_id]["params"]:
            bounds = self.models[model_id]["params"][param_name].get("bounds")
            if bounds:
                return bounds
            
            # Default bounds if none specified
            value = self.models[model_id]["params"][param_name]["value"]
            if value > 0:
                return (value / 10, value * 10)
            elif value < 0:
                return (value * 10, value / 10)
            else:
                return (-1, 1)
        
        return (-1, 1)  # Default fallback
    
    def update_parameter(self, model_id: str, param_name: str, value: float) -> None:
        """Update a parameter value.
        
        Parameters
        ----------
        model_id : str
            ID of the model
        param_name : str
            Name of the parameter
        value : float
            New value for the parameter
        """
        if model_id in self.models and param_name in self.models[model_id]["params"]:
            self.models[model_id]["params"][param_name]["value"] = value
    
    def calculate_model_intensity(self, model_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the model intensity.
        
        Parameters
        ----------
        model_id : str
            ID of the model
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            q values and corresponding intensities
        """
        if model_id not in self.models:
            return self.q_range, np.zeros_like(self.q_range)
        
        model = self.models[model_id]
        kernel = model["kernel"]
        
        # Create parameter dictionary for calculation
        params = {}
        for name, param_info in model["params"].items():
            name = str(name).replace("<", "").replace(">", "")
            params[str(name)] = param_info["value"]
        
        # Apply q range filtering
        q_mask = (self.q_range >= model["q_min"]) & (self.q_range <= model["q_max"])
        q_values = self.q_range[q_mask]
        
        try:
            # Calculate the model
            # Create empty data with resolution
            data = sasmodels.data.empty_data1D(q_values)
            # Create direct model calculator
            calculator = sasmodels.direct_model.DirectModel(data, kernel)
            # Calculate intensity
            intensity = calculator(**params)
            
            return q_values, intensity
        except Exception as e:
            print(f"Error calculating model: {e}")
            # Return empty arrays on error
            return np.array([]), np.array([])
    
    def get_model_inputs(self) -> List[Dict[str, Any]]:
        """Get the model inputs in the format required by AutoSAS.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of model configurations
        """
        result = []
        
        for model_id, model in self.models.items():
            # Convert our internal format to AutoSAS format
            fit_params = {}
            # Group parameters by their base name (for polydispersity)
            param_groups = {}
            
            for name, param_info in model["params"].items():
                # Only include parameter if AutoSAS is enabled
                if param_info.get("autosas", False):
                    # Check if this is a polydispersity parameter
                    if name.endswith(('_pd', '_pd_type', '_pd_n', '_pd_nsigma')):
                        # Get the base parameter name
                        base_name = name.rsplit('_pd', 1)[0]
                        if base_name not in param_groups:
                            param_groups[base_name] = {}
                        
                        # Store the polydispersity parameter
                        if name.endswith('_pd'):
                            param_groups[base_name]['pd'] = param_info["value"]
                        elif name.endswith('_pd_type'):
                            param_groups[base_name]['pd_type'] = param_info["value"]
                        elif name.endswith('_pd_n'):
                            param_groups[base_name]['pd_n'] = param_info["value"]
                        elif name.endswith('_pd_nsigma'):
                            param_groups[base_name]['pd_nsigma'] = param_info["value"]
                    else:
                        # Regular parameter
                        param_dict = {"value": param_info["value"]}
                        
                        # Only include bounds if bounds mode is enabled and bounds exist
                        if (param_info.get("use_bounds", False) and 
                            param_info.get("bounds") is not None):
                            param_dict["bounds"] = param_info["bounds"]
                        
                        fit_params[name] = param_dict
            
            # Add polydispersity parameters to their base parameters
            for base_name, pd_params in param_groups.items():
                if base_name in fit_params:
                    # Only add polydispersity if the base parameter is included
                    for pd_key, pd_value in pd_params.items():
                        fit_params[base_name][pd_key] = pd_value
            
            # Create the model config
            model_config = {
                "name": model["name"],
                "sasmodel": model["sasmodel"],
                "q_min": model["q_min"],
                "q_max": model["q_max"],
                "fit_params": fit_params
            }
            
            result.append(model_config)
        
        return result
    
    def update_q_range(self, model_id: str, q_min: float = None, q_max: float = None) -> None:
        """Update the q range for a model.
        
        Parameters
        ----------
        model_id : str
            ID of the model
        q_min : float, optional
            Minimum q value
        q_max : float, optional
            Maximum q value
        """
        if model_id in self.models:
            if q_min is not None:
                self.models[model_id]["q_min"] = q_min
            if q_max is not None:
                self.models[model_id]["q_max"] = q_max


class AutoSASWidget_View:
    """View component for AutoSASWidget.
    
    Handles the UI elements and their layout.
    """
    
    def __init__(
        self,
        available_models=None,
        data: Optional[xr.Dataset] = None
    ):
        """Initialize the view component.
        
        Parameters
        ----------
        available_models : List[str], optional
            List of available SAS model names
        data : Optional[xr.Dataset]
            Dataset containing experimental scattering data.
        """
        # Containers for UI elements
        self.tabs = widgets.Tab()
        self.dropdown = {}
        self.button = {}
        self.text_input = {}
        self.checkbox = {}
        self.figures = {}
        
        # Available models
        self.available_models = available_models or [
            "sphere", "cylinder", "ellipsoid", "core_shell_sphere",
            "core_shell_cylinder", "power_law"
        ]
        
        # Track the tab IDs
        self.tab_ids = []
        
        # Track parameter controls for each model
        self.parameter_controls = {}
        self.parameter_checkboxes = {}
        
        # Store the dataset
        self.data = data
        
        # Initialize the UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI elements."""
        # Create title
        title = widgets.HTML(
            "<div style='background-color: #f8f9fa; padding: 10px; border-bottom: 2px solid #dee2e6;'>"
            "<h2 style='margin: 0;'>AutoSAS Model Builder</h2>"
            "</div>"
        )
        
        # Create data selection controls if data is provided
        data_controls = None
        if self.data is not None:
            # Get data variables that could be intensity data (1D or 2D arrays)
            data_vars = [var for var, da in self.data.data_vars.items() 
                        if len(da.dims) in [1, 2]]
            data_vars = [var for var, da in self.data.data_vars.items() 
                        if len(da.dims) in [1, 2]]
            # Create data variable dropdown
            self.dropdown["data_var"] = widgets.Dropdown(
                options=data_vars,
                description="Data:",
                value=data_vars[0] if data_vars else None,
                layout=widgets.Layout(width='200px')
            )
            
            # Create q dimension dropdown
            self.dropdown["q_dim"] = widgets.Dropdown(
                options=[],
                description="q dim:",
                layout=widgets.Layout(width='200px')
            )
            
            # Create data index slider
            self.text_input["data_index"] = widgets.IntSlider(
                value=0,
                min=0,
                max=1,
                step=0.1,
                description="Index:",
                #continuous_update=True,  # Add this to update while sliding
                #readout_format='.0f',    # Show as integer
                layout=widgets.Layout(width='300px')
            )
            
            # Update controls when data variable changes
            def update_data_controls(change):
                da = self.data[change["new"]]
                
                # Update q dimension options
                q_dims = [dim for dim in da.dims if 'q' in dim.lower()]
                self.dropdown["q_dim"].options = q_dims
                if q_dims:
                    self.dropdown["q_dim"].value = q_dims[0]
                
                # Update index slider based on non-q dimension
                non_q_dims = [dim for dim in da.dims if 'q' not in dim.lower()]
                if non_q_dims:
                    index_dim = non_q_dims[0]
                    index_size = da.sizes[index_dim]
                    self.text_input["data_index"].max = index_size - 1
                    self.text_input["data_index"].step = 1
                    self.text_input["data_index"].value = 0  # Reset to first index
                
                # Update plots
                for model_id in self.figures:
                    self.update_plot(model_id, [], [])
            
            # Update plot when any control changes
            def update_data_display(*args):
                for model_id in self.figures:
                    self.update_plot(model_id, [], [])
            
            # Connect all data control observers
            self.dropdown["data_var"].observe(update_data_controls, names="value")
            self.dropdown["q_dim"].observe(update_data_display, names="value")
            self.text_input["data_index"].observe(update_data_display, names="value")
            self.button["refresh_data"] = widgets.Button(
                            description="Refresh Data",
                            button_style="info",
                            icon='refresh',
                            layout=widgets.Layout(width='120px', margin='5px 0px 0px 10px')
                        )
            
            # Initialize controls with first data variable
            if data_vars:
                update_data_controls({"new": data_vars[0]})
            
            # Create data controls container
            data_controls = widgets.VBox([
                widgets.HTML("<b>Select Data:</b>"),
                widgets.HBox([
                    widgets.VBox([
                        self.dropdown["data_var"], 
                        self.dropdown["q_dim"],
                        self.text_input["data_index"],
                    ]),
                ], layout=widgets.Layout(margin='5px 0px 15px 10px'))
            ], layout=widgets.Layout(margin='10px 0px', padding='10px',
                                   border='1px solid #eee'))

            # Add observer for refresh button
            def refresh_data(b):
                for model_id in self.figures:
                    self.update_plot(model_id, [], [])
            
            self.button["refresh_data"].on_click(refresh_data)
        
        # Create the model selection dropdown with search
        self.dropdown["model_type"] = widgets.Select(
            options=self.available_models,
            value=self.available_models[0] if self.available_models else "sphere",
            description="Model Type:",
            layout=widgets.Layout(width='300px', height='100px'),
            search=True,  # Enable fuzzy search
            ensure_option=True
        )
        
        # Create add/remove model buttons
        self.button["add_model"] = widgets.Button(
            description="Add Model",
            button_style="success",
            icon='plus',
            layout=widgets.Layout(width='120px')
        )
        
        self.button["remove_model"] = widgets.Button(
            description="Remove Model",
            button_style="danger",
            icon='minus',
            layout=widgets.Layout(width='140px')
        )
        
        # Create export button
        self.button["export"] = widgets.Button(
            description="Export Model Inputs",
            button_style="info",
            icon='download',
            layout=widgets.Layout(width='180px')
        )
        
        # Create notification widget (initially hidden)
        self.notification = widgets.HTML(
            value='',
            layout=widgets.Layout(display='none', margin='5px 0px 0px 0px', 
                                color='green', font_style='italic')
        )
        
        # Create model selection container
        model_selection = widgets.VBox([
            widgets.HTML("<b>Select Model:</b>"),
            self.dropdown["model_type"]
        ], layout=widgets.Layout(margin='10px 0px', padding='10px',
                               border='1px solid #eee'))
        
        # Create buttons container
        buttons_container = widgets.VBox([
            self.button["add_model"],
            self.button["remove_model"],
            widgets.VBox([
                self.button["export"],
                self.notification
            ])
        ], layout=widgets.Layout(margin='20px 0px 0px 20px'))
        
        # Create the top control panel with better spacing
        controls = [model_selection, buttons_container]
        if data_controls is not None:
            controls.insert(0, data_controls)
        top_controls = widgets.HBox(
            controls,
            layout=widgets.Layout(margin='10px 0px 20px 0px', justify_content='flex-start')
        )
        
        # Create the tabs container for models with styling
        self.tabs = widgets.Tab(layout=widgets.Layout(width='100%', min_height='600px'))
        
        # Create the main container
        self.main_container = widgets.VBox([
            title,
            top_controls,
            self.tabs
        ], layout=widgets.Layout(width='1050px', padding='20px'))
    
    def add_model_tab(self, model_id: str, model_name: str) -> None:
        """Add a new tab for a model."""
        # Create plot figure with explicit notebook initialization
        fig = go.FigureWidget(layout={
        })
        fig.update_xaxes(type="log",title="q (Å⁻¹)")
        fig.update_yaxes(type="log",title="Intensity (cm⁻¹)")
        fig.update_layout({
            "legend":dict( yanchor="top", y=0.99, xanchor="right", x=0.01), 
            "margin": dict(l=20, r=0, t=40, b=20),
            "title": f"{model_name} Model",
            "template": "plotly_white",
            "showlegend": False,
            "height": 500,  # Increased height
            "width": 500,   # Increased width
        })
        
        # Add initial trace with better styling
        fig.add_scatter(
            x=[], 
            y=[], 
            mode="lines", 
            line=dict(color="royalblue", width=3),
            name="Model",
            hovertemplate="q: %{x:.3e}<br>I: %{y:.3e}<extra></extra>"
        )
        
        # Add trace for experimental data if available
        if self.data is not None:
            fig.add_scatter(
                x=[], y=[], mode="markers", 
                marker=dict(color="black", size=6),
                name="Data",
                hovertemplate="q: %{x:.3e}<br>I: %{y:.3e}<extra></extra>")
        
        # Store the figure
        self.figures[model_id] = fig
        
        # Create title for parameters section with better styling
        params_title = widgets.HTML(
            "<div style='background-color: #f8f9fa; padding: 10px; border-bottom: 2px solid #dee2e6;'>"
            "<h3 style='margin: 0;'>Model Parameters</h3>"
            "</div>"
        )
        
        # Create filter checkboxes
        filter_title = widgets.HTML("<b>Show Parameters:</b>")
        polydispersity_checkbox = widgets.Checkbox(
            value=False,
            description="Polydispersity",
            indent=False,
            layout=widgets.Layout(width='150px', margin='5px 0px 0px 10px')
        )
        magnetic_checkbox = widgets.Checkbox(
            value=False,
            description="Magnetic",
            indent=False,
            layout=widgets.Layout(width='150px', margin='5px 0px 0px 10px')
        )
        
        # Create filter box
        filter_box = widgets.VBox([
            filter_title,
            widgets.HBox([polydispersity_checkbox, magnetic_checkbox],
                layout=widgets.Layout(margin='5px 0px 15px 10px'))
        ],
            layout=widgets.Layout(
                margin='10px 0px',
                border='1px solid #eee',
                padding='0px 0px 10px 0px',
                min_height='30px',
                flex='none'
            )
        )
        
        # Create q range controls with better organization
        q_range_title = widgets.HTML("<b>q range:</b>")
        q_min_input = widgets.FloatText(
            value=0.001,
            description="q min:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='180px')
        )
        
        q_max_input = widgets.FloatText(
            value=1.0,
            description="q max:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='180px')
        )
        
        q_range_box = widgets.VBox([
            q_range_title,
            widgets.HBox([q_min_input, q_max_input], 
                layout=widgets.Layout(margin='5px 0px 15px 10px')),
        ],
            layout=widgets.Layout(
                margin='10px 0px',
                border='1px solid #eee',
                padding='0px 0px 10px 0px',
                min_height='60px',
                flex='none'
            )
        )
        
        # Create parameters container with better scroll and styling
        params_container = widgets.VBox(
            [params_title, filter_box, q_range_box],
            layout=widgets.Layout(
                width='400px',          # Increased width
                height='550px',         # Match plot height
                overflow_y='scroll',    # Ensure vertical scrolling
                border='1px solid #dee2e6',
                box_shadow='1px 1px 3px rgba(0,0,0,0.1)',
                margin='0px 0px 0px 20px'  # Add margin to separate from plot
            )
        )
        
        # Initialize parameter controls containers
        self.parameter_controls[model_id] = {}
        self.parameter_checkboxes[model_id] = {}
        
        # Create the tab layout with better spacing
        tab_content = widgets.HBox(
            [fig, params_container],
            layout=widgets.Layout(
                justify_content='flex-start',
                padding='20px',
                height='600px',
            )
        )
        
        # Add to tabs
        current_tabs = list(self.tabs.children)
        current_tabs.append(tab_content)
        self.tabs.children = tuple(current_tabs)
        
        # Set the tab title
        self.tabs.set_title(len(current_tabs) - 1, model_name)
        
        # Store the model ID
        self.tab_ids.append(model_id)
        
        # Store q range controls
        self.parameter_controls[model_id]["q_min"] = q_min_input
        self.parameter_controls[model_id]["q_max"] = q_max_input
        
        # Store filter checkboxes
        self.parameter_checkboxes[model_id]["polydispersity_filter"] = polydispersity_checkbox
        self.parameter_checkboxes[model_id]["magnetic_filter"] = magnetic_checkbox
        
        # Add observers for filter checkboxes
        def update_parameter_visibility(change):
            show_polydispersity = polydispersity_checkbox.value
            show_magnetic = magnetic_checkbox.value
            
            # Get all parameter groups (skip title, filter box, and q range box)
            param_groups = list(params_container.children)[3:]
            
            for param_group in param_groups:
                if isinstance(param_group, widgets.VBox):
                    # Get parameter name from title
                    title_html = param_group.children[0].value
                    m = re.search(r'<b>(.*?)</b>', title_html)
                    if m:
                        param_name = m.group(1).strip()
                        
                        # Check if parameter should be visible
                        is_pd_param = '_pd' in param_name
                        is_magnetic_param = (param_name.startswith('up_') or 
                                           '_M0' in param_name or 
                                           '_mphi' in param_name or 
                                           '_mtheta' in param_name)
                        
                        # Update visibility
                        if (is_pd_param and not show_polydispersity) or \
                           (is_magnetic_param and not show_magnetic):
                            param_group.layout.display = 'none'
                        else:
                            param_group.layout.display = None
        
        polydispersity_checkbox.observe(update_parameter_visibility, names='value')
        magnetic_checkbox.observe(update_parameter_visibility, names='value')
    
    def remove_model_tab(self, tab_index: int) -> None:
        """Remove a model tab.
        
        Parameters
        ----------
        tab_index : int
            Index of the tab to remove
        """
        if 0 <= tab_index < len(self.tabs.children):
            # Get the model ID
            model_id = self.tab_ids[tab_index]
            
            # Remove the figure
            if model_id in self.figures:
                del self.figures[model_id]
            
            # Remove parameter controls
            if model_id in self.parameter_controls:
                del self.parameter_controls[model_id]
            
            if model_id in self.parameter_checkboxes:
                del self.parameter_checkboxes[model_id]
            
            # Remove the tab
            current_tabs = list(self.tabs.children)
            del current_tabs[tab_index]
            self.tabs.children = tuple(current_tabs)
            
            # Remove the ID
            del self.tab_ids[tab_index]
    
    def add_parameter_control(
        self, 
        model_id: str, 
        param_name: str, 
        value: Union[float, str],
        bounds: Optional[Tuple[float, float]] = None,
        use_autosas: bool = False,
        use_bounds: bool = False
    ) -> None:
        """Add a control for a parameter.
        
        Parameters
        ----------
        model_id : str
            ID of the model
        param_name : str
            Name of the parameter
        value : Union[float, str]
            Initial value for the parameter
        bounds : Optional[Tuple[float, float]]
            Optional bounds for the parameter
        use_autosas : bool
            Whether to enable AutoSAS for this parameter
        use_bounds : bool
            Whether to enable bounds for this parameter
        """
        # Get the tab index
        tab_index = self.tab_ids.index(model_id)
        
        # Get the parameters container (second item in the HBox)
        params_container = self.tabs.children[tab_index].children[1]
        
        # Check if parameter should be initially visible
        is_pd_param = '_pd' in param_name
        is_magnetic_param = (param_name.startswith('up_') or 
                           '_M0' in param_name or 
                           '_mphi' in param_name or 
                           '_mtheta' in param_name)
        
        show_polydispersity = self.parameter_checkboxes[model_id]["polydispersity_filter"].value
        show_magnetic = self.parameter_checkboxes[model_id]["magnetic_filter"].value
        
        # Determine initial display state
        initial_display = 'none' if ((is_pd_param and not show_polydispersity) or 
                                   (is_magnetic_param and not show_magnetic)) else None
        
        # Create the appropriate input widget based on parameter type
        if param_name.endswith('_pd_type'):
            # Create a dropdown for distribution type
            input_widget = widgets.Dropdown(
                value=value,
                options=bounds if bounds else ["gaussian", "rectangular", "schulz"],
                description="Type:",
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='180px', margin='5px 0px 0px 10px')
            )
        elif param_name.endswith('_pd_n'):
            # Create an integer input for number of points
            input_widget = widgets.IntText(
                value=int(value),
                description="Npts:",
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='180px', margin='5px 0px 0px 10px')
            )
        elif param_name.endswith('_pd_nsigma'):
            # Create a float input for number of sigmas
            input_widget = widgets.FloatText(
                value=float(value),
                description="Nsigmas:",
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='180px', margin='5px 0px 0px 10px')
            )
        elif param_name.endswith('_pd'):
            # Create a float input for polydispersity width
            input_widget = widgets.FloatText(
                value=float(value),
                description="Width:",
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='180px', margin='5px 0px 0px 10px')
            )
        else:
            # Create a standard float input for regular parameters
            input_widget = widgets.FloatText(
                value=float(value),
                description="Value:",
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='180px', margin='5px 0px 0px 10px')
            )
        
        # Create a checkbox for AutoSAS (include in model_inputs)
        autosas_checkbox = widgets.Checkbox(
            value=use_autosas,
            description="AutoSAS",
            indent=False,
            layout=widgets.Layout(width='90px', margin='5px 0px 0px 10px')
        )
        
        # Create a checkbox for bounds/slider
        bounds_checkbox = widgets.Checkbox(
            value=use_bounds,
            description="Bounds",
            indent=False,
            layout=widgets.Layout(width='80px', margin='5px 0px 0px 10px')
        )
        
        # If bounds are enabled, create the slider
        if use_bounds and bounds is not None:
            # Create a row for the parameter with just the value display
            param_row = widgets.HBox([input_widget, autosas_checkbox, bounds_checkbox])
            
            # Create a title for the parameter
            param_title = widgets.HTML(f"<b>{param_name}</b>")
            
            # Create a parameter group
            param_group = widgets.VBox([
                param_title,
                param_row
            ], layout=widgets.Layout(
                margin='10px 0px',
                border='1px solid #eee',
                padding='0px 0px 10px 0px',
                min_height='60px',
                flex='none',
                display=initial_display
            ))
            
            # Add to parameters container
            current_params = list(params_container.children)
            current_params.append(param_group)
            params_container.children = tuple(current_params)
            
            # Store references to the controls
            self.parameter_controls[model_id][param_name] = input_widget
            self.parameter_checkboxes[model_id][param_name] = {
                "autosas": autosas_checkbox,
                "bounds": bounds_checkbox
            }
            
            # Now add the slider
            if len(param_group.children) == 2:
                if value > 0 and bounds[0] > 0:
                    slider_widget = widgets.FloatLogSlider(
                        value=value,
                        base=10,
                        min=np.log10(bounds[0]),
                        max=np.log10(bounds[1]),
                        description='Value',
                        continuous_update=True,
                        layout=widgets.Layout(width='300px')
                    )
                else:
                    slider_widget = widgets.FloatSlider(
                        value=value,
                        min=bounds[0],
                        max=bounds[1],
                        step=(bounds[1]-bounds[0])/100,
                        description='Value',
                        continuous_update=True,
                        layout=widgets.Layout(width='300px')
                    )

                slider_min = widgets.FloatText(
                    value=bounds[0],
                    description='Min:',
                    layout=widgets.Layout(width='200px')
                )
                slider_max = widgets.FloatText(
                    value=bounds[1],
                    description='Max:',
                    layout=widgets.Layout(width='200px')
                )

                slider_row = widgets.VBox([
                    slider_widget, slider_min, slider_max
                ], layout=widgets.Layout(margin='5px 0px'))

                new_children = list(param_group.children) + [slider_row]
                param_group.children = tuple(new_children)

                def update_label(change):
                    input_widget.value = change['new']
                slider_widget.observe(update_label, names='value')

                def update_slider_min(change):
                    if slider_widget.__class__.__name__ == 'FloatSlider':
                        slider_widget.min = change['new']
                    else:
                        slider_widget.min = np.log10(change['new']) if change['new'] > 0 else -3
                slider_min.observe(update_slider_min, names='value')

                def update_slider_max(change):
                    if slider_widget.__class__.__name__ == 'FloatSlider':
                        slider_widget.max = change['new']
                    else:
                        slider_widget.max = np.log10(change['new']) if change['new'] > 0 else 2
                slider_max.observe(update_slider_max, names='value')
        else:
            # Create a row for the parameter
            param_row = widgets.HBox([input_widget, autosas_checkbox, bounds_checkbox])
            
            # Create a title for the parameter
            param_title = widgets.HTML(f"<b>{param_name}</b>")
            
            # Create a parameter group with fixed minimum height to prevent shrinkage
            param_group = widgets.VBox([
                param_title,
                param_row
            ], layout=widgets.Layout(
                margin='10px 0px',
                border='1px solid #eee',
                padding='0px 0px 10px 0px',
                min_height='60px',
                flex='none',
                display=initial_display
            ))
            
            # Add to parameters container
            current_params = list(params_container.children)
            current_params.append(param_group)
            params_container.children = tuple(current_params)
            
            # Store references to the controls
            self.parameter_controls[model_id][param_name] = input_widget
            self.parameter_checkboxes[model_id][param_name] = {
                "autosas": autosas_checkbox,
                "bounds": bounds_checkbox
            }
    
    def get_parameter_control(self, model_id: str, param_name: str) -> widgets.Widget:
        """Get the control for a parameter.
        
        Parameters
        ----------
        model_id : str
            ID of the model
        param_name : str
            Name of the parameter
            
        Returns
        -------
        widgets.Widget
            The control widget
        """
        if model_id in self.parameter_controls and param_name in self.parameter_controls[model_id]:
            return self.parameter_controls[model_id][param_name]
        return None
    
    def get_parameter_checkbox(self, model_id: str, param_name: str) -> widgets.Checkbox:
        """Get the checkbox for a parameter."""
        if model_id in self.parameter_checkboxes and param_name in self.parameter_checkboxes[model_id]:
            # Default to returning the AutoSAS checkbox
            return self.parameter_checkboxes[model_id][param_name]["autosas"]
        return None
    
    def get_bounds_checkbox(self, model_id: str, param_name: str) -> widgets.Checkbox:
        """Get the bounds checkbox for a parameter."""
        if model_id in self.parameter_checkboxes and param_name in self.parameter_checkboxes[model_id]:
            return self.parameter_checkboxes[model_id][param_name]["bounds"]
        return None
    
    def replace_with_slider(self, model_id: str, param_name: str, value: float, bounds: Tuple[float, float]) -> None:
        # Get the tab index and parameters container
        tab_index = self.tab_ids.index(model_id)
        params_container = self.tabs.children[tab_index].children[1]

        # Iterate over parameter groups and extract parameter name from HTML title
        for i, param_group in enumerate(params_container.children):
            if isinstance(param_group, widgets.VBox):
                title_html = param_group.children[0].value
                m = re.search(r'<b>(.*?)</b>', title_html)
                if m and param_name.lower() in m.group(1).strip().lower():
                    # Ensure that the parameter group has only the title and controls row
                    if len(param_group.children) > 2:
                        param_group.children = param_group.children[:2]
                    controls_row = param_group.children[1]
                    if len(param_group.children) == 2:
                        if value > 0 and bounds[0] > 0:
                            slider_widget = widgets.FloatLogSlider(
                                value=value,
                                base=10,
                                min=np.log10(bounds[0]),
                                max=np.log10(bounds[1]),
                                description='Value',
                                #style={'description_width': 'initial'},
                                continuous_update=True,
                                layout=widgets.Layout(width='300px')
                            )
                        else:
                            slider_widget = widgets.FloatSlider(
                                value=value,
                                min=bounds[0],
                                max=bounds[1],
                                step=(bounds[1]-bounds[0])/100,
                                description='Value',
                                #style={'description_width': 'initial'},
                                continuous_update=True,
                                layout=widgets.Layout(width='300px')
                            )

                        slider_min = widgets.FloatText(
                            value=bounds[0],
                            description='Min:',
                            layout=widgets.Layout(width='200px')
                        )
                        slider_max = widgets.FloatText(
                            value=bounds[1],
                            description='Max:',
                            layout=widgets.Layout(width='200px')
                        )

                        slider_row = widgets.VBox([
                            slider_widget, slider_min, slider_max
                        ], layout=widgets.Layout(margin='5px 0px'))

                        new_children = list(param_group.children) + [slider_row]
                        param_group.children = tuple(new_children)

                        self.parameter_controls[model_id][param_name + '_slider'] = slider_widget

                        def update_label(change):
                            controls_row.children[0].value = change['new']
                        slider_widget.observe(update_label, names='value')

                        def update_slider_min(change):
                            if slider_widget.__class__.__name__ == 'FloatSlider':
                                slider_widget.min = change['new']
                            else:
                                slider_widget.min = np.log10(change['new']) if change['new'] > 0 else -3
                        slider_min.observe(update_slider_min, names='value')

                        def update_slider_max(change):
                            if slider_widget.__class__.__name__ == 'FloatSlider':
                                slider_widget.max = change['new']
                            else:
                                slider_widget.max = np.log10(change['new']) if change['new'] > 0 else 2
                        slider_max.observe(update_slider_max, names='value')
                    break
    
    def replace_with_text(self, model_id: str, param_name: str, value: float) -> None:
        tab_index = self.tab_ids.index(model_id)
        params_container = self.tabs.children[tab_index].children[1]

        for i, param_group in enumerate(params_container.children):
            if isinstance(param_group, widgets.VBox):
                title_html = param_group.children[0].value
                m = re.search(r'<b>(.*?)</b>', title_html)
                if m and param_name.lower() in m.group(1).strip().lower():
                    # Get the current checkboxes to preserve their state
                    current_controls = param_group.children[1]
                    autosas_checkbox = None
                    bounds_checkbox = None
                    
                    # Find the checkboxes in the current controls
                    for child in current_controls.children:
                        if isinstance(child, widgets.Checkbox):
                            if child.description == "AutoSAS":
                                autosas_checkbox = child
                            elif child.description == "Bounds":
                                bounds_checkbox = child
                    
                    # If we couldn't find the checkboxes, create new ones
                    if not autosas_checkbox:
                        autosas_checkbox = widgets.Checkbox(
                            value=False,
                            description="AutoSAS",
                            indent=False,
                            layout=widgets.Layout(width='90px', margin='5px 0px 0px 10px')
                        )
                    
                    if not bounds_checkbox:
                        bounds_checkbox = widgets.Checkbox(
                            value=False,
                            description="Bounds",
                            indent=False,
                            layout=widgets.Layout(width='80px', margin='5px 0px 0px 10px')
                        )
                    
                    # Create a new text input
                    text_input = widgets.FloatText(
                        value=value,
                        description="Value:",
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(width='180px', margin='5px 0px 0px 10px')
                    )
                    
                    # Create a new controls row with all elements
                    new_controls_row = widgets.HBox([text_input, autosas_checkbox, bounds_checkbox])
                    
                    # Remove any slider row if it exists
                    param_group.children = (param_group.children[0], new_controls_row)
                    
                    # Update the control references
                    self.parameter_controls[model_id][param_name] = text_input
                    self.parameter_checkboxes[model_id][param_name] = {
                        "autosas": autosas_checkbox,
                        "bounds": bounds_checkbox
                    }
                    
                    # # Add observer for the text input
                    # def update_label(change):
                    #     text_input.value = f"{change['new']:.4g}"
                    # text_input.observe(update_label, names='value')
                    break
    
    def update_plot(self, model_id: str, q: np.ndarray, intensity: np.ndarray) -> None:
        """Update the plot for a model.
        
        Parameters
        ----------
        model_id : str
            ID of the model
        q : np.ndarray
            Array of q values
        intensity : np.ndarray
            Array of intensity values
        """
        if model_id in self.figures:
            fig = self.figures[model_id]
            with fig.batch_update():
                if len(q) > 0 and len(intensity) > 0:  # Only update if we have data
                    fig.data[0].x = q
                    fig.data[0].y = intensity
                
                # Update experimental data if available
                if self.data is not None and len(fig.data) > 1:
                    data_var = self.dropdown["data_var"].value
                    q_dim = self.dropdown["q_dim"].value
                    if data_var and q_dim:
                        da = self.data[data_var]
                        
                        # Get q values from the selected q dimension
                        q_data = da[q_dim].values if q_dim in da.dims else None
                        
                        # Get intensity data
                        if len(da.dims) == 1:
                            # 1D data
                            i_data = da.values
                        else:
                            # 2D data with index
                            non_q_dims = [dim for dim in da.dims if dim != q_dim]
                            if non_q_dims:
                                index_dim = non_q_dims[0]
                                index = int(self.text_input["data_index"].value)
                                i_data = da.isel({index_dim: index}).values
                            else:
                                i_data = None

                        if q_data is not None and i_data is not None:
                            fig.data[1].x = q_data
                            fig.data[1].y = i_data
                        else:
                            fig.data[1].x = []
                            fig.data[1].y = []
    
    def clear_tabs(self) -> None:
        """Clear all tabs."""
        # Clear tab IDs
        self.tab_ids = []
        
        # Clear parameter controls
        self.parameter_controls = {}
        self.parameter_checkboxes = {}
        
        # Clear figures
        self.figures = {}
        
        # Clear tabs
        self.tabs.children = ()

# Example usage in a Jupyter notebook
if __name__ == "__main__":
    # This code will run when the file is executed directly
    # Helpful for testing and demonstration
    
    print("""
    # Example of how to use AutoSASWidget in a notebook:
    
    from AFL.double_agent.AutoSASWidget import AutoSASWidget
    
    # Create the widget
    widget = AutoSASWidget()
    
    # Display it
    widget.run()
    
    # Later, after configuring models, you can get the model inputs:
    model_inputs = widget.get_model_inputs()
    
    # And use them with AutoSAS:
    # from AFL.double_agent.AutoSAS import AutoSAS
    # autosas = AutoSAS(..., model_inputs=model_inputs)
    """) 