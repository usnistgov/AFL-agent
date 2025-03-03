import ipywidgets
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import copy
import uuid
import json
import sasmodels.data
import sasmodels.core
import ipywidgets as widgets
from IPython.display import display, JSON

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
        default_models: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize the AutoSASWidget.
        
        Parameters
        ----------
        q_range : Optional[np.ndarray]
            Array of q values to use for model visualization.
            If None, a default range will be created.
        default_models : Optional[List[Dict[str, Any]]]
            List of model configurations to initialize with.
            If None, starts with an empty model list.
        """
        # Set up default q_range if not provided
        if q_range is None:
            self.q_range = np.logspace(-3, 0, 100)
        else:
            self.q_range = q_range
            
        # Create the model and view
        self.model = AutoSASWidget_Model(q_range=self.q_range, default_models=default_models)
        self.view = AutoSASWidget_View(available_models=self.model.available_sasmodels)
        
        # Connect model and view
        self._setup_callbacks()
    
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
    
    def _setup_parameter_controls(self, model_id):
        """Set up parameter controls for a model."""
        # Get parameters for the model
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
        
        # Create controls for each parameter
        for param_name, param_info in params.items():
            # Create a slider for this parameter with the current value
            value = param_info["value"]
            param_name = str(param_name)
            
            # Create the parameter control in the view
            self.view.add_parameter_control(
                model_id, 
                param_name,
                value,
                param_info.get("bounds", None)
            )
            
            # Set up callback for value changes
            self.view.get_parameter_control(model_id, param_name).observe(
                lambda change, mid=model_id, pname=param_name: 
                    self.parameter_change_callback(change, mid, pname),
                names="value"
            )
            
            # Set up callback for checkbox changes
            checkbox = self.view.get_parameter_checkbox(model_id, param_name)
            if checkbox:
                checkbox.observe(
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
        # Get current controls
        control = self.view.get_parameter_control(model_id, param_name)
        
        # If checkbox is checked, we want to enable slider mode
        if change["new"]:
            # Get current value
            current_value = control.value
            
            # Get default bounds for this parameter
            bounds = self.model.get_parameter_bounds(model_id, param_name)
            
            # Replace text input with slider
            self.view.replace_with_slider(model_id, param_name, current_value, bounds)
            
            # Set up callback for slider
            slider = self.view.get_parameter_control(model_id, param_name)
            slider.observe(
                lambda change, mid=model_id, pname=param_name: 
                    self.parameter_change_callback(change, mid, pname),
                names="value"
            )
        else:
            # Replace slider with text input
            current_value = control.value
            self.view.replace_with_text(model_id, param_name, current_value)
            
            # Set up callback for text input
            text = self.view.get_parameter_control(model_id, param_name)
            text.observe(
                lambda change, mid=model_id, pname=param_name: 
                    self.parameter_change_callback(change, mid, pname),
                names="value"
            )
    
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
            List of model configurations that can be passed to AutoSAS.
        """
        return self.model.get_model_inputs()
    
    def run(self):
        """Display the widget interface."""
        display(self.view.main_container)
    
    def export_callback(self, b):
        """Handle export button click."""
        # Get model inputs
        model_inputs = self.get_model_inputs()
        
        # Display the model inputs as formatted JSON
        display(JSON(model_inputs))
        
        # Show a message about usage with AutoSAS
        display(widgets.HTML(
            "<p><b>How to use:</b> Copy the above JSON data and use it as the model_inputs parameter for AutoSAS.</p>"
            "<p>Example:<br><code>autosas = AutoSAS(..., model_inputs=model_inputs)</code></p>"
        ))
    
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
        default_models: Optional[List[Dict[str, Any]]] = None
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
        self.available_sasmodels = self._get_available_sasmodels()
        
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
        
        # Convert the fit_params format to our internal format
        params = {}
        for name, param_info in model_config["fit_params"].items():
            params[name] = {
                "value": param_info["value"],
                "bounds": param_info.get("bounds", None),
                "fixed": "bounds" not in param_info
            }
        
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
            for name, param_info in model["params"].items():
                param_dict = {"value": param_info["value"]}
                
                # Only include bounds if the parameter is not fixed
                if not param_info["fixed"] and param_info.get("bounds") is not None:
                    param_dict["bounds"] = param_info["bounds"]
                
                fit_params[name] = param_dict
            
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
    
    def __init__(self, available_models=None):
        """Initialize the view component.
        
        Parameters
        ----------
        available_models : List[str], optional
            List of available SAS model names
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
        
        # Initialize the UI
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI elements."""
        # Create title
        title = widgets.HTML("<h2>AutoSAS Model Builder</h2>")
        
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
        
        # Create the top control panel with better spacing
        top_controls = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<b>Select Model:</b>"),
                self.dropdown["model_type"]
            ]),
            widgets.VBox([
                self.button["add_model"],
                self.button["remove_model"],
                self.button["export"]
            ], layout=widgets.Layout(margin='20px 0px 0px 20px'))
        ], layout=widgets.Layout(margin='10px 0px 20px 0px', justify_content='flex-start'))
        
        # Create the tabs container for models with styling
        self.tabs = widgets.Tab(layout=widgets.Layout(width='100%', min_height='600px'))
        
        # Create the main container
        self.main_container = widgets.VBox([
            title,
            top_controls,
            self.tabs
        ], layout=widgets.Layout(width='100%', padding='20px'))
    
    def add_model_tab(self, model_id: str, model_name: str) -> None:
        """Add a new tab for a model."""
        # Create plot figure with explicit notebook initialization
        fig = go.FigureWidget(layout={
            "height": 600,  # Increased height
            "width": 800,   # Increased width
            "title": f"{model_name} Model",
            "xaxis_title": "q (Å⁻¹)",
            "yaxis_title": "Intensity (cm⁻¹)",
            "template": "plotly_white",
            "margin": {"l": 60, "r": 20, "t": 60, "b": 60},
            "showlegend": True
        })
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        
        # Add initial trace with better styling
        fig.add_scatter(
            x=[], 
            y=[], 
            mode="lines", 
            line=dict(color="royalblue", width=3),
            name="Model",
            hovertemplate="q: %{x:.3e}<br>I: %{y:.3e}<extra></extra>"
        )
        
        # Store the figure
        self.figures[model_id] = fig
        
        # Create title for parameters section with better styling
        params_title = widgets.HTML(
            "<div style='background-color: #f8f9fa; padding: 10px; border-bottom: 2px solid #dee2e6;'>"
            "<h3 style='margin: 0;'>Model Parameters</h3>"
            "</div>"
        )
        
        # Create q range controls with better organization
        q_range_title = widgets.HTML("<b>Q Range:</b>")
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
                layout=widgets.Layout(margin='5px 0px 15px 10px'))
        ])
        
        # Create parameters container with better scroll and styling
        params_container = widgets.VBox(
            [params_title, q_range_box],
            layout=widgets.Layout(
                width='400px',          # Increased width
                height='600px',         # Match plot height
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
                padding='20px'
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
        value: float,
        bounds: Optional[Tuple[float, float]] = None
    ) -> None:
        """Add a control for a parameter."""
        # Get the tab index
        tab_index = self.tab_ids.index(model_id)
        
        # Get the parameters container (second item in the HBox)
        params_container = self.tabs.children[tab_index].children[1]
        
        # Create parameter section title
        param_title = widgets.HTML(
            f"<div style='margin-top: 10px; padding: 5px; background-color: #f8f9fa;'>"
            f"<b>{param_name}</b>"
            f"</div>"
        )
        
        # Create a text input for the parameter with better styling
        text_input = widgets.FloatText(
            value=value,
            description="Value:",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='180px', margin='5px 0px 0px 10px')
        )
        
        # Create a checkbox for enabling/disabling slider mode
        checkbox = widgets.Checkbox(
            value=False,
            description="Use Slider",
            indent=False,
            layout=widgets.Layout(width='100px', margin='5px 0px 0px 10px')
        )
        
        # Create value label to show parameter value
        value_label = widgets.Label(
            value=f"Current: {value:.4g}",
            layout=widgets.Layout(width='120px', margin='5px 0px 0px 10px')
        )
        
        # Create a parameter group with fixed minimum height to prevent shrinkage
        param_group = widgets.VBox([
            param_title,
            widgets.HBox([text_input, checkbox, value_label])
        ], layout=widgets.Layout(
            margin='10px 0px',
            border='1px solid #eee',
            padding='0px 0px 10px 0px',
            min_height='60px',
            flex='none'
        ))
        
        # Add to parameters container
        current_params = list(params_container.children)
        current_params.append(param_group)
        params_container.children = tuple(current_params)
        
        # Store references to controls
        self.parameter_controls[model_id][param_name] = text_input
        self.parameter_checkboxes[model_id][param_name] = checkbox
    
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
        """Get the checkbox for a parameter.
        
        Parameters
        ----------
        model_id : str
            ID of the model
        param_name : str
            Name of the parameter
            
        Returns
        -------
        widgets.Checkbox
            The checkbox widget
        """
        if model_id in self.parameter_checkboxes and param_name in self.parameter_checkboxes[model_id]:
            return self.parameter_checkboxes[model_id][param_name]
        return None
    
    def replace_with_slider(
        self, 
        model_id: str, 
        param_name: str, 
        value: float, 
        bounds: Tuple[float, float]
    ) -> None:
        """Replace a text input with a slider.
        
        Parameters
        ----------
        model_id : str
            ID of the model
        param_name : str
            Name of the parameter
        value : float
            Current value of the parameter
        bounds : Tuple[float, float]
            Bounds for the slider (min, max)
        """
        # Get the tab index
        tab_index = self.tab_ids.index(model_id)
        
        # Get the parameters container (second item in the HBox)
        params_container = self.tabs.children[tab_index].children[1]
        
        # Find the current row for this parameter
        for i, child in enumerate(params_container.children):
            if isinstance(child, widgets.HBox) and child.children[0].description == param_name:
                # Create a slider
                slider = widgets.FloatLogSlider(
                    value=value,
                    base=10,
                    min=np.log10(bounds[0]) if bounds[0] > 0 else -3,
                    max=np.log10(bounds[1]) if bounds[1] > 0 else 2,
                    description=param_name,
                    style={'description_width': 'initial'},
                    continuous_update=True,
                    layout=widgets.Layout(width='180px')
                ) if value > 0 and bounds[0] > 0 else widgets.FloatSlider(
                    value=value,
                    min=bounds[0],
                    max=bounds[1],
                    step=(bounds[1]-bounds[0])/100,
                    description=param_name,
                    style={'description_width': 'initial'},
                    continuous_update=True,
                    layout=widgets.Layout(width='180px')
                )
                
                # Get the existing checkbox
                checkbox = child.children[1]
                
                # Get or create the value label
                value_label = child.children[2] if len(child.children) > 2 else widgets.Label()
                value_label.value = f"{value:.4g}"
                
                # Create new row
                new_row = widgets.HBox([slider, checkbox, value_label])
                
                # Update the parameters container
                current_params = list(params_container.children)
                current_params[i] = new_row
                params_container.children = tuple(current_params)
                
                # Update reference
                self.parameter_controls[model_id][param_name] = slider
                
                # Add observer to update label when slider changes
                def update_label(change):
                    value_label.value = f"{change['new']:.4g}"
                
                slider.observe(update_label, names='value')
                break
    
    def replace_with_text(self, model_id: str, param_name: str, value: float) -> None:
        """Replace a slider with a text input.
        
        Parameters
        ----------
        model_id : str
            ID of the model
        param_name : str
            Name of the parameter
        value : float
            Current value of the parameter
        """
        # Get the tab index
        tab_index = self.tab_ids.index(model_id)
        
        # Get the parameters container (second item in the HBox)
        params_container = self.tabs.children[tab_index].children[1]
        
        # Find the current row for this parameter
        for i, child in enumerate(params_container.children):
            if isinstance(child, widgets.HBox) and child.children[0].description == param_name:
                # Create a text input
                text_input = widgets.FloatText(
                    value=value,
                    description=param_name,
                    style={'description_width': 'initial'}
                )
                
                # Get the existing checkbox
                checkbox = child.children[1]
                
                # Get or create the value label
                value_label = child.children[2] if len(child.children) > 2 else widgets.Label()
                value_label.value = f"{value:.4g}"
                
                # Create new row
                new_row = widgets.HBox([text_input, checkbox, value_label])
                
                # Update the parameters container
                current_params = list(params_container.children)
                current_params[i] = new_row
                params_container.children = tuple(current_params)
                
                # Update reference
                self.parameter_controls[model_id][param_name] = text_input
                
                # Add observer to update label when text input changes
                def update_label(change):
                    value_label.value = f"{change['new']:.4g}"
                
                text_input.observe(update_label, names='value')
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
                    
                    # Ensure axes stay in log mode
                    fig.layout.xaxis.type = 'log'
                    fig.layout.yaxis.type = 'log'
                    
                    # # Update ranges if we have valid data
                    # if np.any(q > 0) and np.any(intensity > 0):
                    #     x_min = np.min(q[q > 0])
                    #     x_max = np.max(q)
                    #     y_min = np.min(intensity[intensity > 0])
                    #     y_max = np.max(intensity)
                    #     
                    #     ## Add some padding in log space
                    #     #fig.layout.xaxis.range = [np.log10(x_min) - 0.1, np.log10(x_max) + 0.1]
                    #     #fig.layout.yaxis.range = [np.log10(y_min) - 0.1, np.log10(y_max) + 0.1]
                
                # Ensure plot maintains log-log scale and desired size
                # fig.layout.xaxis.type = 'log'
                # fig.layout.yaxis.type = 'log'
                fig.layout.height = 600  # match the model parameters box height
    
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