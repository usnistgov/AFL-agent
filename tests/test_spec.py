import json

import pytest
import xarray as xr

from AFL.double_agent.spec import encode_spec, resolve_spec, import_class, class_path
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.Pipeline import Pipeline


class ClassArgOp(PipelineOp):
    """Minimal op taking a class-valued argument."""

    def __init__(self, input_variable, output_variable, factory, name="ClassArgOp"):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)
        self.factory = factory

    def calculate(self, dataset):
        self.output[self.output_variable] = dataset[self.input_variable].copy()
        return self


def test_encode_type_to_spec():
    assert encode_spec(xr.Dataset) == {"class": "xarray.core.dataset.Dataset"}


def test_encode_spec_dict_with_type():
    spec = encode_spec({"class": xr.Dataset, "args": {"attrs": {"a": 1}}})
    assert spec == {"class": "xarray.core.dataset.Dataset", "args": {"attrs": {"a": 1}}}
    json.dumps(spec)


def test_encode_passthrough():
    for v in [1, "x", None, [1, 2], {"a": 1}]:
        assert encode_spec(v) == v


def test_class_path_rejects_main_and_lambda():
    with pytest.raises(TypeError):
        class_path(lambda: None)
    Local = type("Local", (), {"__module__": "__main__"})
    with pytest.raises(TypeError):
        class_path(Local)


def test_import_class_and_resolve():
    assert import_class("xarray.core.dataset.Dataset") is xr.Dataset
    ds = resolve_spec({"class": "xarray.core.dataset.Dataset", "args": {"attrs": {"a": 1}}})
    assert isinstance(ds, xr.Dataset) and ds.attrs["a"] == 1
    assert isinstance(resolve_spec("xarray.core.dataset.Dataset"), xr.Dataset)
    assert isinstance(resolve_spec(xr.Dataset), xr.Dataset)
    ds = resolve_spec({"class": xr.Dataset}, attrs={"b": 2})
    assert ds.attrs["b"] == 2


def test_import_class_factory_function():
    f = import_class("AFL.double_agent.spec.encode_spec")
    assert f is encode_spec


def test_pipelineop_accepts_class_argument():
    op = ClassArgOp("a", "b", factory=xr.Dataset)
    assert op.factory == {"class": "xarray.core.dataset.Dataset"}
    assert op._stored_args["factory"] == {"class": "xarray.core.dataset.Dataset"}
    json.dumps(op.to_json())

    op.factory = {"class": xr.DataArray, "args": {"data": [1]}}
    assert op.factory["class"] == "xarray.core.dataarray.DataArray"


def test_pipelineop_still_rejects_non_json():
    with pytest.raises(TypeError):
        ClassArgOp("a", "b", factory=xr.Dataset())


def test_pipeline_json_roundtrip_with_class_argument(tmp_path):
    with Pipeline(name="p") as p:
        ClassArgOp("a", "b", factory={"class": xr.Dataset, "args": {"attrs": {"k": 1}}})
    f = tmp_path / "p.json"
    p.write_json(str(f))
    p2 = Pipeline.read_json(str(f))
    assert p2[0].factory == {"class": "xarray.core.dataset.Dataset", "args": {"attrs": {"k": 1}}}
    built = resolve_spec(p2[0].factory)
    assert isinstance(built, xr.Dataset) and built.attrs["k"] == 1
