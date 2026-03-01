import json
import pathlib
from typing import Any, Dict

import xarray as xr
from importlib.resources import files
from jinja2 import Template

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
from AFL.double_agent.util import listify


class AgentWebAppMixin:
    """Reusable web app endpoints and static asset bindings for agent drivers."""

    static_dirs = {
        "js": pathlib.Path(__file__).parent / "apps" / "pipeline_builder" / "js",
        "img": pathlib.Path(__file__).parent / "apps" / "pipeline_builder" / "img",
        "css": pathlib.Path(__file__).parent / "apps" / "pipeline_builder" / "css",
        "input_builder_js": pathlib.Path(__file__).parent / "apps" / "input_builder" / "js",
        "input_builder_css": pathlib.Path(__file__).parent / "apps" / "input_builder" / "css",
    }

    @staticmethod
    def _render_pipeline_builder_html() -> str:
        template_path = files("AFL.double_agent.apps").joinpath("pipeline_builder").joinpath("pipeline_builder.html")
        template = Template(template_path.read_text())
        return template.render()

    @staticmethod
    def _render_input_builder_html() -> str:
        template_path = files("AFL.double_agent.apps").joinpath("input_builder").joinpath("input_builder.html")
        template = Template(template_path.read_text())
        return template.render()

    def setup_app_links(self) -> None:
        if self.useful_links is None:
            self.useful_links = {
                "Pipeline Builder": "/pipeline_builder",
                "Input Builder": "/input_builder",
            }
            return

        self.useful_links["Pipeline Builder"] = "/pipeline_builder"
        self.useful_links["Input Builder"] = "/input_builder"

    @Driver.unqueued(render_hint="html")
    def pipeline_builder(self, **kwargs):
        """Serve the pipeline builder HTML interface."""
        return self._render_pipeline_builder_html()

    @Driver.unqueued(render_hint="html")
    def input_builder(self, **kwargs):
        """Serve the input builder HTML interface."""
        return self._render_input_builder_html()

    @Driver.unqueued(render_hint="precomposed_svg")
    def plot_pipeline(self, **kwargs):
        if self.pipeline is not None:
            return mpl_plot_to_bytes(self.pipeline.draw(), format="svg")
        return None

    @Driver.unqueued(render_hint="html")
    def last_result(self, **kwargs):
        return self.last_results._repr_html_()

    @Driver.unqueued(render_hint="netcdf")
    def download_last_result(self, **kwargs):
        return xarray_to_bytes(self.last_results)

    @Driver.unqueued(render_hint="precomposed_png")
    def plot_operation(self, operation, **kwargs):
        try:
            operation = int(operation)
        except ValueError:
            pass
        if self.pipeline is not None:
            if isinstance(operation, str):
                return mpl_plot_to_bytes(self.pipeline.search(operation).plot(), format="png")
            if isinstance(operation, int):
                return mpl_plot_to_bytes(self.pipeline[operation].plot(), format="png")
            return None
        return None

    @Driver.unqueued()
    def get_tiled_input_config(self, **kwargs):
        """Return current tiled_input_groups configuration."""
        return {
            "status": "success",
            "config": self.config.get("tiled_input_groups", []),
        }

    @Driver.unqueued()
    def set_tiled_input_config(self, config: str = None, **kwargs):
        """Update tiled_input_groups configuration."""
        try:
            if config is None:
                return {"status": "error", "message": "config parameter required"}

            if isinstance(config, str):
                config_list = json.loads(config)
            else:
                config_list = config

            if not isinstance(config_list, list):
                return {"status": "error", "message": "config must be a list"}

            for i, group_cfg in enumerate(config_list):
                if not isinstance(group_cfg, dict):
                    return {"status": "error", "message": f"Group {i} must be a dictionary"}
                required_keys = ["concat_dim", "variable_prefix", "entry_ids"]
                for key in required_keys:
                    if key not in group_cfg:
                        return {"status": "error", "message": f"Group {i} missing required key: {key}"}
                if not isinstance(group_cfg["entry_ids"], list):
                    return {"status": "error", "message": f"Group {i} entry_ids must be a list"}

            self.config["tiled_input_groups"] = config_list
            return {
                "status": "success",
                "message": f"Saved {len(config_list)} group(s)",
                "config": config_list,
            }
        except json.JSONDecodeError as e:
            return {"status": "error", "message": f"Invalid JSON: {str(e)}"}
        except Exception as e:
            return {"status": "error", "message": f"Error saving config: {str(e)}"}

    @Driver.unqueued()
    def test_fetch_entry(self, entry_id: str = None, **kwargs):
        """Test fetching a single entry from tiled to validate it exists."""
        if entry_id is None:
            return {"status": "error", "message": "entry_id parameter required"}

        try:
            client = self._get_tiled_client()
            if isinstance(client, dict) and client.get("status") == "error":
                return client

            if entry_id not in client:
                return {"status": "error", "message": f'Entry "{entry_id}" not found in tiled'}

            item = client[entry_id]
            metadata = dict(item.metadata) if hasattr(item, "metadata") else {}

            try:
                from tiled.client.xarray import DatasetClient

                if isinstance(item, DatasetClient):
                    dataset = item.read(optimize_wide_table=False)
                    dims_info = dict(dataset.sizes)
                    data_vars = list(dataset.data_vars)
                else:
                    dims_info = {}
                    data_vars = []
            except Exception:
                dims_info = {}
                data_vars = []

            return {
                "status": "success",
                "entry_id": entry_id,
                "metadata": metadata,
                "dims": dims_info,
                "data_vars": data_vars,
            }
        except Exception as e:
            return {"status": "error", "message": f"Error testing entry: {str(e)}"}

    @Driver.unqueued()
    def pipeline_ops(self, **kwargs):
        """Return metadata for available PipelineOps."""
        import importlib

        agent_driver_module = importlib.import_module("AFL.double_agent.AgentDriver")
        strict = agent_driver_module._parse_strict_flag(kwargs.get("strict"))
        return agent_driver_module.get_pipeline_ops(strict=strict)

    @Driver.unqueued()
    def current_pipeline(self, **kwargs):
        """Return the currently loaded pipeline as JSON."""
        if self.pipeline is None:
            return None
        connections = self._make_connections(self.pipeline)
        return {"ops": [op.to_json() for op in self.pipeline], "connections": connections}

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
        connections = self._make_connections(pipeline)
        return {"ops": [op.to_json() for op in pipeline], "connections": connections}

    def _make_connections(self, pipeline: Pipeline):
        connections = []
        output_var_to_op_indices = {}
        for i, op in enumerate(pipeline.ops):
            for output_var in listify(op.output_variable):
                if output_var is None:
                    continue
                if output_var not in output_var_to_op_indices:
                    output_var_to_op_indices[output_var] = []
                output_var_to_op_indices[output_var].append(i)

        for target_index, target_op in enumerate(pipeline.ops):
            for input_var in listify(target_op.input_variable):
                if input_var is not None and input_var in output_var_to_op_indices:
                    for source_index in output_var_to_op_indices[input_var]:
                        if source_index != target_index:
                            connections.append(
                                {
                                    "source_index": source_index,
                                    "target_index": target_index,
                                    "variable": input_var,
                                }
                            )
        return connections

    @Driver.unqueued()
    def save_prefab(self, name: str, pipeline: str = "[]", overwrite: bool = True, **kwargs):
        """Save a pipeline (sent from the UI) as a prefab JSON file."""
        from AFL.double_agent.Pipeline import Pipeline as _Pipeline
        from AFL.double_agent.PipelineOp import PipelineOp
        from AFL.double_agent.prefab import save_prefab as _save_prefab

        try:
            ops_def = json.loads(pipeline) if isinstance(pipeline, str) else pipeline
        except Exception:
            return {"status": "error", "message": "Invalid pipeline JSON."}

        try:
            pipeline_ops = [PipelineOp.from_json(op) for op in ops_def]
            pipeline_obj = _Pipeline(name=name, ops=pipeline_ops)
            path = _save_prefab(pipeline_obj, name=name, overwrite=overwrite)
            return {"status": "success", "path": path}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @Driver.unqueued()
    def build_pipeline(self, ops: str = "[]", name: str = "Pipeline", **kwargs):
        """Construct a pipeline from JSON and return the serialized form."""
        import importlib

        agent_driver_module = importlib.import_module("AFL.double_agent.AgentDriver")
        return agent_driver_module.build_pipeline_from_json(ops, name)

    @Driver.unqueued()
    def analyze_pipeline(self, ops: str = "[]", **kwargs):
        """Analyze pipeline operations and return connectivity information."""
        from AFL.double_agent.PipelineOp import PipelineOp

        try:
            ops_list = json.loads(ops)
        except (json.JSONDecodeError, TypeError):
            return {"connections": [], "errors": []}

        pipeline_ops = []
        errors = []
        for i, op_def in enumerate(ops_list):
            try:
                op = PipelineOp.from_json(op_def)
                pipeline_ops.append(op)
            except Exception as e:
                op_name = op_def["args"].get("name", f"Operation {i}")
                op_class = op_def.get("class", "Unknown class")
                error_msg = f"Failed to instantiate '{op_name}' ({op_class}): {str(e)}"
                errors.append(
                    {
                        "operation_index": i,
                        "operation_name": op_name,
                        "operation_class": op_class,
                        "error": str(e),
                        "error_message": error_msg,
                    }
                )

        if errors:
            return {
                "connections": [],
                "errors": errors,
                "status": "error",
                "message": f"Failed to instantiate {len(errors)} operation(s). Pipeline analysis stopped.",
            }

        if not pipeline_ops:
            return {"connections": [], "errors": []}

        connections = []
        output_var_to_op_indices = {}
        for i, op in enumerate(pipeline_ops):
            for output_var in listify(op.output_variable):
                if output_var is not None:
                    if output_var not in output_var_to_op_indices:
                        output_var_to_op_indices[output_var] = []
                    output_var_to_op_indices[output_var].append(i)

        for target_index, target_op in enumerate(pipeline_ops):
            for input_var in listify(target_op.input_variable):
                if input_var is not None and input_var in output_var_to_op_indices:
                    for source_index in output_var_to_op_indices[input_var]:
                        if source_index != target_index:
                            connections.append(
                                {
                                    "source_index": source_index,
                                    "target_index": target_index,
                                    "variable": input_var,
                                }
                            )
        return {"connections": connections, "errors": []}

    def _assemble_group(self, group_cfg: Dict[str, Any]) -> xr.Dataset:
        """Assemble a single group of entries from tiled."""
        concat_dim = group_cfg.get("concat_dim")
        variable_prefix = group_cfg.get("variable_prefix", "")
        entry_ids = group_cfg.get("entry_ids", [])
        if not entry_ids:
            raise ValueError(f"Group with concat_dim '{concat_dim}' has no entry_ids")
        return self.tiled_concat_datasets(
            entry_ids=entry_ids,
            concat_dim=concat_dim,
            variable_prefix=variable_prefix,
        )

    @Driver.queued()
    def assemble_input_from_tiled(self, **kwargs):
        """Assemble input dataset from tiled entries based on configured groups."""
        groups = self.config.get("tiled_input_groups", [])
        if not groups:
            return {
                "status": "error",
                "message": "No tiled_input_groups configured. Use Input Builder to configure.",
            }

        assembled = []
        for i, group_cfg in enumerate(groups):
            try:
                group_ds = self._assemble_group(group_cfg)
                assembled.append(group_ds)
            except Exception as e:
                return {
                    "status": "error",
                    "message": f'Failed to assemble group {i} (concat_dim={group_cfg.get("concat_dim", "unknown")}): {str(e)}',
                }

        if not assembled:
            return {"status": "error", "message": "No datasets assembled"}

        merged = xr.merge(assembled, compat="override")
        self.input = merged.reset_coords()
        html_repr = (
            self.input._repr_html_()
            if hasattr(self.input, "_repr_html_")
            else f"<pre>{str(self.input)}</pre>"
        )
        return {
            "status": "success",
            "dims": dict(self.input.sizes),
            "data_vars": list(self.input.data_vars),
            "coords": list(self.input.coords),
            "html": html_repr,
        }

    @Driver.unqueued()
    def check_predict_ready(self, **kwargs):
        """Check if predict can be called successfully."""
        if self.pipeline is None:
            return {"ready": False, "error": "No pipeline loaded"}
        if self.input is None:
            return {"ready": False, "error": "No input assembled"}

        required_vars = self.pipeline.input_variables()
        required_vars = [v for v in required_vars if "generator" not in v.lower()]
        available_vars = list(self.input.data_vars) + list(self.input.coords)
        missing = [v for v in required_vars if v not in available_vars]

        if missing:
            return {
                "ready": False,
                "error": f"Missing input variables: {missing}",
                "required": required_vars,
                "available": available_vars,
            }
        return {"ready": True, "required": required_vars, "available": available_vars}
