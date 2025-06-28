import json
import pathlib
import uuid
from importlib.resources import files
from typing import Any, Dict, Optional

import xarray as xr
from AFL.automation.APIServer.Driver import Driver  # type: ignore
from jinja2 import Template

from .dataset_plugins import load_plugins


def get_dataset_builder_html() -> str:
    template_path = (
        files("AFL.double_agent.driver_templates")
        .joinpath("dataset_builder")
        .joinpath("dataset_builder.html")
    )
    template = Template(template_path.read_text())
    return template.render()


class DatasetBuilderDriver(Driver):
    """Driver serving the dataset builder web UI and handling uploads."""

    defaults: Dict[str, Any] = {}
    defaults["save_path"] = "/tmp"

    static_dirs = {
        "js": pathlib.Path(__file__).parent / "driver_templates" / "dataset_builder" / "js",
        "css": pathlib.Path(__file__).parent / "driver_templates" / "dataset_builder" / "css",
    }

    def __init__(
        self, name: str = "DatasetBuilderDriver", overrides: Optional[Dict[str, Any]] = None
    ):
        Driver.__init__(self, name=name, defaults=self.gather_defaults(), overrides=overrides)
        self.dataset: Optional[xr.Dataset] = None
        self.plugins = load_plugins()

    @Driver.unqueued(render_hint="html")
    def dataset_builder(self, **kwargs):
        return get_dataset_builder_html()

    @Driver.unqueued()
    def plugin_list(self, **kwargs):
        return list(self.plugins.keys())

    @Driver.unqueued()
    def reset(self, **kwargs):
        self.dataset = None

    @Driver.unqueued()
    def get_dataset(self, **kwargs):
        return self.dataset

    @Driver.unqueued(render_hint="html")
    def get_dataset_html(self, **kwargs):
        if self.dataset is None:
            return "<p>No dataset</p>"
        return self.dataset._repr_html_()

    @Driver.unqueued()
    def upload_data(
        self,
        plugin: str,
        filename: str,
        file_bytes: bytes,
        dims: str = "{}",
        coords: str = "{}",
        **kwargs,
    ):
        if plugin not in self.plugins:
            raise ValueError(f"Unknown plugin {plugin}")

        loader = self.plugins[plugin]

        tmp_path = pathlib.Path(self.config["save_path"]) / f"upload_{uuid.uuid4()}"
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        try:
            data = loader(str(tmp_path))
            dims_map = json.loads(dims)
            coords_map = json.loads(coords)
            if dims_map:
                data = data.rename(dims_map)
            for name, values in coords_map.items():
                data = data.assign_coords({name: values})

            if self.dataset is None:
                self.dataset = data
            else:
                try:
                    self.dataset = xr.concat([self.dataset, data], dim=list(data.dims)[0])
                except Exception as e:
                    return {"error": str(e)}

            return {"dims": list(self.dataset.dims), "data_vars": list(self.dataset.data_vars)}
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
