import pathlib
import uuid
import json
import inspect
import importlib
import pkgutil
from typing import Optional, Dict, Any, List, get_type_hints, Union

import xarray as xr

from AFL.automation.APIServer.Driver import Driver  # type: ignore
from AFL.automation.shared.utilities import mpl_plot_to_bytes,xarray_to_bytes
from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify

from importlib.resources import files
from jinja2 import Template


def _get_parameter_types(cls) -> Dict[str, str]:
    """Extract parameter types from class constructor type annotations.
    
    Parameters
    ----------
    cls : class
        The class to extract parameter types from
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping parameter names to their type strings
    """
    param_types = {}
    try:
        # Get type hints from __init__ method
        type_hints = get_type_hints(cls.__init__)
        for param_name, param_type in type_hints.items():
            if param_name != 'return' and param_name != 'self':
                # Convert type to string representation
                type_str = str(param_type)
                
                # Handle typing module types by checking origin and args
                origin = getattr(param_type, '__origin__', None)
                args = getattr(param_type, '__args__', ())
                
                # Detect dictionary types (including Union[Dict, None] for Optional)
                if (param_type == dict or 
                    origin == dict or 
                    'Dict' in type_str or 
                    'dict' in type_str or
                    (origin == type(Union) and any('Dict' in str(arg) or arg == dict for arg in args))):
                    param_types[param_name] = 'dict'
                # Detect list types
                elif (param_type == list or 
                      origin == list or 
                      'List' in type_str or 
                      'list' in type_str or
                      (origin == type(Union) and any('List' in str(arg) or arg == list for arg in args))):
                    param_types[param_name] = 'list'
                # Basic types
                elif param_type == str or type_str == "<class 'str'>":
                    param_types[param_name] = 'str'
                elif param_type == int or type_str == "<class 'int'>":
                    param_types[param_name] = 'int'
                elif param_type == float or type_str == "<class 'float'>":
                    param_types[param_name] = 'float'
                elif param_type == bool or type_str == "<class 'bool'>":
                    param_types[param_name] = 'bool'
                else:
                    # For complex types, just store the simplified string representation
                    simplified = type_str.replace("<class '", "").replace("'>", "")
                    # Clean up typing module references
                    simplified = simplified.replace("typing.", "")
                    param_types[param_name] = simplified
                    
    except Exception as e:
        # If type hint extraction fails, log but don't crash
        print(f"Warning: Could not extract type hints from {cls.__name__}: {e}")
    
    return param_types


def _collect_pipeline_ops() -> List[Dict[str, Any]]:
    """Gather metadata for all available :class:`PipelineOp` subclasses."""
    ops: List[Dict[str, Any]] = []
    package = importlib.import_module("AFL.double_agent")
    for modinfo in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{modinfo.name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, PipelineOp) and obj is not PipelineOp:
                sig = inspect.signature(obj.__init__)
                params = {
                    k: (v.default if v.default is not inspect._empty else None)
                    for k, v in sig.parameters.items()
                    if k != "self"
                }
                
                # Detect input/output variable parameters
                input_params = []
                output_params = []
                
                for param_name in params.keys():
                    # Parameters that represent input variables
                    if (param_name.endswith('_variable') and 'input' in param_name) or \
                       param_name == 'input_variable' or param_name == 'input_variables':
                        input_params.append(param_name)
                    # Parameters that represent output variables or prefixes
                    elif (param_name.endswith('_variable') and 'output' in param_name) or \
                         param_name == 'output_variable' or param_name == 'output_variables' or \
                         param_name == 'output_prefix':
                        output_params.append(param_name)
                
                # Extract parameter types from type annotations
                param_types = _get_parameter_types(obj)

                ops.append(
                    {
                        "name": name,
                        "module": module.__name__,
                        "fqcn": f"{module.__name__}.{name}",
                        "parameters": params,
                        "input_params": input_params,
                        "output_params": output_params,
                        "param_types": param_types,
                        "docstring": inspect.getdoc(obj) or "",
                    }
                )
    ops.sort(key=lambda o: o["name"])
    return ops


def get_pipeline_ops() -> List[Dict[str, Any]]:
    """Return metadata describing available pipeline operations."""
    return _collect_pipeline_ops()


def build_pipeline_from_ops(ops: List[Dict[str, Any]], name: str = "Pipeline") -> Dict[str, Any]:
    """Create a pipeline from a list of operation JSON dictionaries."""
    pipeline_ops = [PipelineOp.from_json(op) for op in ops]
    pipeline = Pipeline(name=name, ops=pipeline_ops)
    return {"pipeline": [op.to_json() for op in pipeline]}


def build_pipeline_from_json(ops_json: str, name: str = "Pipeline") -> Dict[str, Any]:
    """Helper that accepts a JSON string of operations."""
    try:
        ops = json.loads(ops_json)
    except json.JSONDecodeError:
        ops = []
    return build_pipeline_from_ops(ops, name)


def get_pipeline_builder_html() -> str:
    """Return the HTML for the pipeline builder UI."""
    template_path = files('AFL.double_agent.driver_templates').joinpath('pipeline_builder.html')
    template = Template(template_path.read_text())
    html = template.render()
    return html


class DoubleAgentDriver(Driver):
    """
    Persistent Config
    -----------------
    save_path: str
        path to directory where data will be serialized to
    """

    defaults = {}
    defaults["save_path"] = "/home/AFL/"
    defaults["pipeline"] = {}

    def __init__(
        self,
        name: str = "DoubleAgentDriver",
        overrides: Optional[Dict[str, Any]] = None,
    ):
        Driver.__init__(
            self, name=name, defaults=self.gather_defaults(), overrides=overrides
        )
        self.app = None
        self.name = name


        if self.config["pipeline"]:
            assert "name" in self.config["pipeline"], "Pipeline name in config is required"
            assert "ops" in self.config["pipeline"], "Pipeline ops in config are required"
            assert "description" in self.config["pipeline"], "Pipeline description in config is required"

            self. pipeline = Pipeline(
                name=self.config["pipeline"]["name"],
                ops=[PipelineOp.from_json(op) for op in self.config["pipeline"]["ops"]],
                description=self.config["pipeline"]["description"],
            )
        else:
            self.pipeline: Optional[Pipeline] = None


        self.input: Optional[xr.Dataset] = None
        self.results: Dict[str, xr.Dataset] = dict()

        self.useful_links = {
            "Pipeline Builder": "/pipeline_builder"
        }


    def status(self):
        status = []
        if self.input:
            status.append(f'Input Dims: {self.input.sizes}')
        if self.pipeline:
            status.append(f'Pipeline loaded with {len(self.pipeline.ops)} operations')
        return status
        

    def initialize_input(self, db_uuid: str) -> None:
        """
        Set the initial input data to be evaluated in the `double_agent.Pipeline`

        Parameters
        ----------
        db_uuid: str
            Dropbox UUID to retrieve `xarray.Dataset` from. The Dataset should be deposited using `Client.deposit_obj`
            in interactive mode in order to obtain the uuid of the deposited item. See example below

        Example
        -------
        ```python
        from AFL.automation.APIServer.Client import Client
        import xarray as xr

        client = Client('localhost',port=5053)
        client.login('User')
        db_uuid = client.deposit_obj(obj=xr.Dataset(),interactive=True)['return_val']
        client.enqueue(task_name='initialize_input',db_uuid=db_uuid)
        ```
        """
        self.input = self.retrieve_obj(db_uuid)

    def initialize_pipeline(self, db_uuid: str = None, pipeline: List[Dict[str, Any]] = None, name: str = "Pipeline") -> None:
        """
        Set the `double_agent.Pipeline` to outline

        Parameters
        ----------
        db_uuid: str, optional
            Dropbox UUID to retrieve the `double_agent.Pipeline` from. The Dataset should be deposited using
            `Client.deposit_obj` in interactive mode in order to obtain the uuid of the deposited item. 
            Either this or `pipeline` must be provided.
        pipeline: List[Dict[str, Any]], optional
            List of pipeline operations in JSON format to construct the pipeline from.
            Either this or `db_uuid` must be provided.
        name: str, default="Pipeline"
            Name for the pipeline when constructing from operations list.

        Example
        -------
        Using dropbox:
        ```python
        from AFL.automation.APIServer.Client import Client
        from AFL.double_agent import *

        client = Client('localhost',port=5053)
        client.login('User')
        db_uuid = client.deposit_obj(obj=Pipeline(),interactive=True)['return_val']
        client.enqueue(task_name='initialize_pipeline',db_uuid=db_uuid)
        ```
        
        Using operations list:
        ```python
        ops = [{'class': 'AFL.double_agent.SomeOp', 'args': {...}}]
        client.enqueue(task_name='initialize_pipeline', pipeline=ops)
        ```
        """
        if db_uuid is not None and pipeline is not None:
            raise ValueError("Cannot specify both db_uuid and pipeline. Use one or the other.")
        
        if db_uuid is None and pipeline is None:
            raise ValueError("Must specify either db_uuid or pipeline.")
        
        if db_uuid is not None:
            # Load pipeline from dropbox
            self.pipeline = self.retrieve_obj(db_uuid)
            self.config["pipeline"] = self.pipeline.to_dict()
        else:
            # Construct pipeline from operations list
            pipeline_ops = [PipelineOp.from_json(op) for op in pipeline]
            self.pipeline = Pipeline(name=name, ops=pipeline_ops)
            self.config["pipeline"] = self.pipeline.to_dict()

    def append(self, db_uuid: str, concat_dim: str) -> None:
        """

        Parameters
        ----------
        db_uuid: str
            Dropbox UUID to retrieve `xarray.Dataset` from. The Dataset should be deposited using `Client.deposit_obj`
            in interactive mode in order to obtain the uuid of the deposited item. See example below

        concat_dim: str
            `xarray` dimension in input dataset to concatenate to

        """
        if self.input is None:
            raise ValueError(
                'Must set "input" Dataset client.deposit_obj and then DoubleAgentDriver.initialize'
            )

        next_sample = self.retrieve_obj(db_uuid)

        if self.input is None:
            self.input = next_sample
        else:
            self.input = xr.concat(
                [self.input, next_sample], dim=concat_dim, data_vars="minimal"
            )

    @Driver.unqueued(render_hint = 'precomposed_svg')
    def plot_pipeline(self,**kwargs):
        if self.pipeline is not None:
            return mpl_plot_to_bytes(self.pipeline.draw(),format='svg')
        else:
            return None

    @Driver.unqueued(render_hint = 'html')
    def last_result(self,**kwargs):
        return self.last_results._repr_html_()

    @Driver.unqueued(render_hint = 'netcdf')
    def download_last_result(self,**kwargs):
        return xarray_to_bytes(self.last_results)
    
    @Driver.unqueued(render_hint = 'precomposed_png')
    def plot_operation(self,operation,**kwargs):
        try:
            operation = int(operation)
        except ValueError:
            pass
        if self.pipeline is not None:
            if isinstance(operation,str):
                return mpl_plot_to_bytes(self.pipeline.search(operation).plot(),format='png')
            elif isinstance(operation,int):
                return mpl_plot_to_bytes(self.pipeline[operation].plot(),format='png')
            else:
                return None
        else:
            return None

    @Driver.unqueued(render_hint='html')
    def pipeline_builder(self, **kwargs):
        """Serve the pipeline builder HTML interface."""
        return get_pipeline_builder_html()

    @Driver.unqueued()
    def pipeline_ops(self, **kwargs):
        """Return metadata for available PipelineOps."""
        return get_pipeline_ops()

    @Driver.unqueued()
    def current_pipeline(self, **kwargs):
        """Return the currently loaded pipeline as JSON."""
        if self.pipeline is None:
            return None
        connections = self._make_connections(self.pipeline)
        return {
            'ops': [op.to_json() for op in self.pipeline],
            'connections': connections
        }

    @Driver.unqueued()
    def prefab_names(self, **kwargs):
        """List available prefabricated pipelines."""
        from AFL.double_agent.prefab import list_prefabs
        return list_prefabs(display_table=False)

    @Driver.unqueued()
    def load_prefab(self, name: str, **kwargs):
        """Load a prefabricated pipeline and return its JSON with connectivity information."""
        from AFL.double_agent.prefab import load_prefab
        
        pipeline = load_prefab(name)
        self.config["pipeline"] = pipeline.to_dict()

        connections = self._make_connections(pipeline)

        return {
            'ops': [op.to_json() for op in pipeline],
            'connections': connections
        }

    def _make_connections(self, pipeline: Pipeline):
        # Create connections between operations based on variable matching
        connections = []
        
        # Create a mapping of output variables to lists of operation indices (one-to-many)
        output_var_to_op_indices = {}
        for i, op in enumerate(pipeline.ops):
            for output_var in listify(op.output_variable):
                if output_var is not None:
                    if output_var not in output_var_to_op_indices:
                        output_var_to_op_indices[output_var] = []
                    output_var_to_op_indices[output_var].append(i)
        
        # Find connections where input variables match output variables
        for target_index, target_op in enumerate(pipeline.ops):
            for input_var in listify(target_op.input_variable):
                if input_var is not None and input_var in output_var_to_op_indices:
                    # Connect to ALL operations that produce this output variable
                    for source_index in output_var_to_op_indices[input_var]:
                        if source_index != target_index:  # Avoid self-loops
                            connections.append({
                                'source_index': source_index,
                                'target_index': target_index,
                                'variable': input_var
                            })

        return connections

    @Driver.unqueued()
    def save_prefab(self, name: str, pipeline: str = "[]", overwrite: bool = True, **kwargs):
        """Save a pipeline (sent from the UI) as a prefab JSON file.

        Parameters
        ----------
        name : str
            Desired filename (without .json) for the prefab.
        pipeline : str
            JSON-encoded list of operation dictionaries from the UI.
        overwrite : bool, default=True
            Whether to overwrite an existing prefab of the same name.
        """
        import json as _json
        from AFL.double_agent.PipelineOp import PipelineOp
        from AFL.double_agent.Pipeline import Pipeline as _Pipeline
        from AFL.double_agent.prefab import save_prefab as _save_prefab

        try:
            ops_def = _json.loads(pipeline) if isinstance(pipeline, str) else pipeline
        except Exception:
            return {
                'status': 'error',
                'message': 'Invalid pipeline JSON.'
            }

        try:
            pipeline_ops = [PipelineOp.from_json(op) for op in ops_def]
            pipeline_obj = _Pipeline(name=name, ops=pipeline_ops)
            path = _save_prefab(pipeline_obj, name=name, overwrite=overwrite)
            return {
                'status': 'success',
                'path': path
            }
        except Exception as exc:
            return {
                'status': 'error',
                'message': str(exc)
            }

    @Driver.unqueued()
    def build_pipeline(self, ops: str = "[]", name: str = "Pipeline", **kwargs):
        """Construct a pipeline from JSON and return the serialized form."""
        return build_pipeline_from_json(ops, name)

    @Driver.unqueued()
    def analyze_pipeline(self, ops: str = "[]", **kwargs):
        """Analyze pipeline operations and return connectivity information."""
        from AFL.double_agent.PipelineOp import PipelineOp
        from AFL.double_agent.util import listify
        import json
        
        # Parse the operations from JSON string
        try:
            ops_list = json.loads(ops)
        except (json.JSONDecodeError, TypeError):
            return {'connections': []}
        
        # Create PipelineOp instances from the operation definitions
        pipeline_ops = []
        for op_def in ops_list:
            try:
                op = PipelineOp.from_json(op_def)
                pipeline_ops.append(op)
            except Exception as e:
                # If we can't instantiate an op, skip it for connectivity analysis
                continue
        
        if not pipeline_ops:
            return {'connections': []}
        
        # Create connections between operations based on variable matching
        connections = []
        
        # Create a mapping of output variables to lists of operation indices (one-to-many)
        output_var_to_op_indices = {}
        for i, op in enumerate(pipeline_ops):
            for output_var in listify(op.output_variable):
                if output_var is not None:
                    if output_var not in output_var_to_op_indices:
                        output_var_to_op_indices[output_var] = []
                    output_var_to_op_indices[output_var].append(i)
        
        # Find connections where input variables match output variables
        for target_index, target_op in enumerate(pipeline_ops):
            for input_var in listify(target_op.input_variable):
                if input_var is not None and input_var in output_var_to_op_indices:
                    # Connect to ALL operations that produce this output variable
                    for source_index in output_var_to_op_indices[input_var]:
                        if source_index != target_index:  # Avoid self-loops
                            connections.append({
                                'source_index': source_index,
                                'target_index': target_index,
                                'variable': input_var
                            })
        
        return {'connections': connections}

    def reset_results(self):
        self.results = dict()

    def predict(
        self,
        deposit: bool = True,
        save_to_disk: bool = True,
        sample_uuid: Optional[str] = None,
        AL_campaign_name: Optional[str] = None,
    ) -> str:
        """
        Evaluate the pipeline set with `.initialize_pipeline`.

        Parameters
        ----------
        deposit: bool
            If True, the `xarray.Dataset` resulting from the `Pipeline` calculation will be placed in this `APIServers`
            dropbox for retrieval and the `db_uuid` will be returned.

        save_to_disk: bool
            If True, the `xarray.Dataset` resulting from the `Pipeline` calculation will be serialized to disk in
            NetCDF format.

        sample_uuid: Optional[str]
            Optionally provide a sample uuid to tag the calculation with

        AL_campaign_name
            Optionally provide an AL campaign name to tag the calculation with

        """
        if (self.pipeline is None) or (self.input is None):
            raise ValueError(
                """Cannot predict without a pipeline and input loaded! Use client.set_driver_object to upload an """
                """Pipeline.Pipeline and an xr.Dataset to this APIServer. You currently have: \n"""
                f"""DoubleAgentDriver.pipeline = {self.pipeline}\n"""
                f"""DoubleAgentDriver.input = {self.input}\n"""
            )
        if sample_uuid is None:
            sample_uuid = 'SAM-'+str(uuid.uuid4())

        ag_uid = "AG-" + str(uuid.uuid4())
        self.results[ag_uid] = self.pipeline.calculate(self.input)

        self.results[ag_uid].attrs['sample_uuid'] = sample_uuid
        self.results[ag_uid].attrs['ag_uuid'] = ag_uid
        self.results[ag_uid].attrs['AL_campaign_name'] = AL_campaign_name

        if save_to_disk:
            path = (
                pathlib.Path(self.config["save_path"])
                / f"{AL_campaign_name}_SAM-{str(sample_uuid)[-6:]}_AG-{ag_uid[-6:]}.nc"
            )
            self.results[ag_uid].to_netcdf(path)

        if deposit:
            self.deposit_obj(self.results[ag_uid], uid=ag_uid)
        
        self.last_results = self.results[ag_uid]
        
        return ag_uid

_OVERRIDE_MAIN_MODULE_NAME = 'DoubleAgentDriver'
if __name__ == '__main__':
    from AFL.automation.shared.launcher import *
