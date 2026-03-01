import pathlib
import uuid
import json
import inspect
import importlib
import copy
import hashlib
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple, get_type_hints, Union

import xarray as xr

try:
    from AFL.automation.APIServer.Driver import Driver  # type: ignore
    from AFL.automation.shared.utilities import mpl_plot_to_bytes, xarray_to_bytes
except ModuleNotFoundError as exc:
    # Allow unit tests to import this module in environments where AFL-automation
    # is not installed. Runtime server behavior still requires AFL-automation.
    if exc.name and exc.name.startswith("AFL.automation"):
        class Driver:  # type: ignore[override]
            @staticmethod
            def unqueued(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator

            @staticmethod
            def queued(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator

            def __init__(self, *args, **kwargs):
                pass

            def gather_defaults(self):
                return getattr(self, "defaults", {})

        def mpl_plot_to_bytes(*args, **kwargs):  # type: ignore[no-redef]
            raise RuntimeError("mpl_plot_to_bytes requires AFL-automation to be installed.")

        def xarray_to_bytes(*args, **kwargs):  # type: ignore[no-redef]
            raise RuntimeError("xarray_to_bytes requires AFL-automation to be installed.")
    else:
        raise
from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.AgentWebAppMixin import AgentWebAppMixin


LOGGER = logging.getLogger(__name__)
_DISCOVERY_LOCK = threading.Lock()
_PIPELINE_OPS_MEM_CACHE: Optional[Dict[str, Any]] = None


def _cache_path() -> pathlib.Path:
    env_path = os.environ.get("AFL_PIPELINE_OPS_CACHE_PATH")
    if env_path:
        return pathlib.Path(env_path).expanduser()
    return pathlib.Path.home() / ".cache" / "afl-double-agent" / "pipeline_ops_manifest.json"


def _candidate_module_files() -> List[pathlib.Path]:
    module_dir = pathlib.Path(__file__).parent
    excluded = {
        "__init__.py",
        "_version.py",
        "AgentDriver.py",
        "util.py",
    }

    module_files: List[pathlib.Path] = []
    for path in sorted(module_dir.glob("*.py")):
        if path.name in excluded or path.name.startswith("_"):
            continue

        # Cheap pre-filter to avoid importing modules that cannot define PipelineOps.
        try:
            if "PipelineOp" not in path.read_text(encoding="utf-8"):
                continue
        except Exception:
            # If the pre-filter fails, keep the module candidate for safety.
            pass
        module_files.append(path)
    return module_files


def _module_signature(module_files: List[pathlib.Path]) -> str:
    hasher = hashlib.sha256()
    for path in module_files:
        stat = path.stat()
        hasher.update(str(path).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
    return hasher.hexdigest()


def _parse_strict_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_disk_cache(expected_signature: str) -> Optional[Dict[str, Any]]:
    cache_file = _cache_path()
    if not cache_file.exists():
        return None
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None

    if cached.get("signature") != expected_signature:
        return None
    if "ops" not in cached or "warnings" not in cached or "generated_at" not in cached:
        return None
    return cached


def _save_disk_cache(payload: Dict[str, Any]) -> None:
    cache_file = _cache_path()
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix="pipeline_ops_", suffix=".json", dir=str(cache_file.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(payload, tmp)
        pathlib.Path(tmp_name).replace(cache_file)
    finally:
        try:
            pathlib.Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _build_warning(module_name: str, stage: str, error: Exception) -> Dict[str, str]:
    return {
        "module": module_name,
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
    }


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


def _collect_pipeline_ops(module_files: List[pathlib.Path], strict: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Gather metadata for all available :class:`PipelineOp` subclasses."""
    ops: List[Dict[str, Any]] = []
    warnings: List[Dict[str, str]] = []

    for module_path in module_files:
        module_name = f"AFL.double_agent.{module_path.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            warning = _build_warning(module_name, "import", e)
            warnings.append(warning)
            LOGGER.warning(
                "Skipping module '%s': failed to import (%s: %s)",
                module_name,
                type(e).__name__,
                e,
            )
            if strict:
                raise RuntimeError(
                    f"PipelineOp discovery failed while importing '{module_name}': {type(e).__name__}: {e}"
                ) from e
            continue

        try:
            members = inspect.getmembers(module, inspect.isclass)
        except Exception as e:
            warning = _build_warning(module_name, "inspect", e)
            warnings.append(warning)
            LOGGER.warning(
                "Skipping module '%s': failed to inspect members (%s: %s)",
                module_name,
                type(e).__name__,
                e,
            )
            if strict:
                raise RuntimeError(
                    f"PipelineOp discovery failed while inspecting '{module_name}': {type(e).__name__}: {e}"
                ) from e
            continue

        for name, obj in members:
            if obj.__module__ != module.__name__:
                continue
            try:
                if not (issubclass(obj, PipelineOp) and obj is not PipelineOp):
                    continue
            except TypeError:
                # issubclass can raise TypeError for some edge cases
                continue
            
            try:
                sig = inspect.signature(obj.__init__)
                params = {
                    k: (v.default if v.default is not inspect._empty else None)
                    for k, v in sig.parameters.items()
                    if k != "self"
                }
                
                # Detect required parameters (those without defaults)
                required_params = []
                for k, v in sig.parameters.items():
                    if k != "self" and v.default is inspect._empty:
                        required_params.append(k)
                
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
                        "required_params": required_params,
                        "input_params": input_params,
                        "output_params": output_params,
                        "param_types": param_types,
                        "docstring": inspect.getdoc(obj) or "",
                    }
                )
            except Exception as e:
                warning = _build_warning(module_name, "metadata", e)
                warnings.append(warning)
                LOGGER.warning(
                    "Skipping PipelineOp '%s' from '%s': failed to extract metadata (%s: %s)",
                    name,
                    module.__name__,
                    type(e).__name__,
                    e,
                )
                if strict:
                    raise RuntimeError(
                        f"PipelineOp discovery failed for class '{name}' in '{module_name}': {type(e).__name__}: {e}"
                    ) from e
                continue

    ops.sort(key=lambda o: o["name"])
    return ops, warnings


def get_pipeline_ops(strict: bool = False) -> Dict[str, Any]:
    """Return metadata describing available pipeline operations with cache metadata."""
    start = time.perf_counter()
    module_files = _candidate_module_files()
    signature = _module_signature(module_files)

    if not strict:
        with _DISCOVERY_LOCK:
            global _PIPELINE_OPS_MEM_CACHE

            if _PIPELINE_OPS_MEM_CACHE and _PIPELINE_OPS_MEM_CACHE.get("signature") == signature:
                result = copy.deepcopy(_PIPELINE_OPS_MEM_CACHE)
                result["cache"]["source"] = "memory"
                result["cache"]["duration_ms"] = int((time.perf_counter() - start) * 1000)
                result.pop("signature", None)
                return result

            disk_cached = _load_disk_cache(signature)
            if disk_cached is not None:
                result = {
                    "ops": disk_cached["ops"],
                    "warnings": disk_cached["warnings"],
                    "cache": {
                        "source": "disk",
                        "generated_at": disk_cached["generated_at"],
                        "signature": disk_cached["signature"],
                        "duration_ms": int((time.perf_counter() - start) * 1000),
                    },
                    "signature": disk_cached["signature"],
                }
                _PIPELINE_OPS_MEM_CACHE = copy.deepcopy(result)
                result.pop("signature", None)
                return result

    ops, warnings = _collect_pipeline_ops(module_files, strict=strict)
    generated_at = datetime.now(timezone.utc).isoformat()
    duration_ms = int((time.perf_counter() - start) * 1000)

    result = {
        "ops": ops,
        "warnings": warnings,
        "cache": {
            "source": "fresh",
            "generated_at": generated_at,
            "signature": signature,
            "duration_ms": duration_ms,
        },
        "signature": signature,
    }

    if not strict:
        cache_payload = {
            "ops": ops,
            "warnings": warnings,
            "generated_at": generated_at,
            "signature": signature,
        }
        with _DISCOVERY_LOCK:
            _PIPELINE_OPS_MEM_CACHE = copy.deepcopy(result)
            try:
                _save_disk_cache(cache_payload)
            except Exception as exc:
                LOGGER.warning("Failed to write pipeline ops cache: %s: %s", type(exc).__name__, exc)

    result.pop("signature", None)
    return result


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


class DoubleAgentDriver(AgentWebAppMixin, Driver):
    """
    Persistent Config
    -----------------
    save_path: str
        path to directory where data will be serialized to
    """

    defaults = {}
    defaults["save_path"] = "/home/AFL/"
    defaults["pipeline"] = {}
    defaults["tiled_input_groups"] = []  # List[Dict] with concat_dim, variable_prefix, entry_ids

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
        self.last_results: Optional[xr.Dataset] = None

        self.setup_app_links()
        


    def status(self):
        status = []
        if 'mock_mode' in self.config:
            status.append(f'mock_mode: {self.config["mock_mode"]}')
        if self.input:
            status.append(f'Input Dims: {self.input.sizes}')
        if self.pipeline:
            status.append(f'Pipeline loaded with {len(self.pipeline.ops)} operations')
        return status
        

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

    @Driver.queued()
    def predict(
        self,
        deposit: bool = False,
        save_to_disk: bool = False,
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
        self.assemble_input_from_tiled()

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
        self.last_results = self.pipeline.calculate(self.input)

        self.last_results.attrs['sample_uuid'] = sample_uuid
        self.last_results.attrs['ag_uuid'] = ag_uid
        self.last_results.attrs['AL_campaign_name'] = AL_campaign_name

        if save_to_disk:
            path = (
                pathlib.Path(self.config["save_path"])
                / f"{AL_campaign_name}_SAM-{str(sample_uuid)[-6:]}_AG-{ag_uid[-6:]}.nc"
            )
            self.last_results.to_netcdf(path)

        if deposit:
            self.deposit_obj(self.last_results, uid=ag_uid)
        
        
        return self.last_results

_OVERRIDE_MAIN_MODULE_NAME = 'DoubleAgentDriver'
if __name__ == '__main__':
    from AFL.automation.shared.launcher import *
