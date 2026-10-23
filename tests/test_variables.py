import copy
import functools
import json
import pickle

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from AFL.double_agent.variables import (
    UnboundVariableError,
    Variable,
    bind_dataset,
    decode_variables,
    find_variables,
    resolve_variables,
)


def make_dataset():
    return xr.Dataset(
        {
            "composition": (("sample", "component"), np.arange(6.0).reshape(3, 2)),
            "labels": (("sample",), np.array([0, 1, 1])),
            "flag": (("sample",), np.array([True, False, True])),
            "names": (("sample",), np.array(["a", "b", "c"])),
        }
    )


def test_unbound_variable_raises_clear_error():
    v = Variable("composition")
    with pytest.raises(UnboundVariableError, match="bind_dataset"):
        v.value
    with pytest.raises(UnboundVariableError):
        v.shape


def test_resolution_kinds():
    ds = make_dataset()
    with bind_dataset(ds, {"manifold": "M"}):
        assert isinstance(Variable("composition").value, torch.Tensor)
        assert Variable("composition").value.dtype == torch.get_default_dtype()
        assert Variable("flag").value.dtype == torch.get_default_dtype()
        assert isinstance(Variable("names").value, np.ndarray)
        assert isinstance(Variable("composition").as_array().value, np.ndarray)
        assert isinstance(Variable("composition").raw().value, xr.DataArray)
        assert Variable("composition").as_tensor(dtype=torch.float32).value.dtype == torch.float32
        assert Variable("manifold").value == "M"  # op-provided
        assert Variable("dataset").value is ds
        with pytest.raises(KeyError, match="Available"):
            Variable("missing").value
        with pytest.raises(TypeError, match="model ref"):
            Variable("names").as_model().value


def test_model_ref_is_loaded(tmp_path):
    pytest.importorskip("gpytorch")
    from AFL.double_agent.torch_io import clear_cache, save_model

    clear_cache()
    model = torch.nn.Linear(2, 1)
    ref = save_model(model, str(tmp_path), args={"in_features": 2, "out_features": 1})
    ds = xr.Dataset({"model": ((), ref)})
    with bind_dataset(ds):
        loaded = Variable("model").value
        assert isinstance(loaded, torch.nn.Linear)
        assert Variable("model").as_model().value is loaded
        # the proxy forwards attribute access and calls to the loaded model
        assert Variable("model").in_features == 2
        out = Variable("model")(torch.zeros(1, 2))
        assert out.shape == (1, 1)
    clear_cache()


def test_proxy_forwards_tensor_behaviour():
    ds = make_dataset()
    x = Variable("composition")
    with bind_dataset(ds):
        assert x.shape == (3, 2)
        assert x[0, 1].item() == 1.0
        assert len(x) == 3
        np.testing.assert_allclose(np.asarray(x), ds["composition"].values)
        torch.testing.assert_close(torch.cat([torch.zeros(1, 2), x]).sum(), torch.tensor(15.0))
        assert torch.stack([x, x]).shape == (2, 3, 2)
        torch.testing.assert_close((2 * x - x).sum(), torch.tensor(15.0))
        torch.testing.assert_close((x + 1).sum(), torch.tensor(21.0))
        torch.testing.assert_close((1 + x).sum(), torch.tensor(21.0))
        torch.testing.assert_close((x**2).sum(), torch.tensor(55.0))
        torch.testing.assert_close((-x).sum(), torch.tensor(-15.0))
        assert torch.cdist(torch.zeros(1, 2), x).shape == (1, 3)
        torch.testing.assert_close(x.reshape(-1)[-1], torch.tensor(5.0))
        assert float(Variable("labels")[1]) == 1.0


def test_resolve_and_decode_containers():
    ds = make_dataset()
    tree = {"a": Variable("labels"), "b": [Variable("composition"), 3], "c": "plain"}
    with bind_dataset(ds):
        out = resolve_variables(tree)
    assert isinstance(out["a"], torch.Tensor) and isinstance(out["b"][0], torch.Tensor)
    assert out["b"][1] == 3 and out["c"] == "plain"

    encoded = {"a": Variable("labels").to_json(), "b": [Variable("x").as_tensor(dtype=torch.float32).to_json()]}
    json.dumps(encoded)
    decoded = decode_variables(encoded)
    assert isinstance(decoded["a"], Variable) and decoded["a"].name == "labels"
    assert decoded["b"][0].kind == "tensor" and decoded["b"][0].dtype == torch.float32


def test_variable_is_immutable_copyable_and_picklable():
    v = Variable("composition", kind="tensor")
    with pytest.raises(AttributeError):
        v.name = "other"
    assert copy.deepcopy(v) is v
    restored = pickle.loads(pickle.dumps(v))
    assert restored.name == "composition" and restored.kind == "tensor"
    assert repr(v) == "Variable('composition', kind='tensor')"


MODULE_LEVEL = Variable("from_globals")


def helper(X, a, b=Variable("from_default")):
    return X


def test_find_variables_in_callables():
    closed = Variable("from_closure")
    fn = lambda X: helper(X, closed, MODULE_LEVEL)  # noqa: E731
    found = find_variables(fn)
    # closure, referenced global, and the default of the global function the lambda calls
    assert set(found) == {"from_closure", "from_globals", "from_default"}

    assert set(find_variables(helper)) == {"from_default"}
    part = functools.partial(helper, Variable("from_partial"), b=closed)
    assert set(find_variables(part)) == {"from_partial", "from_closure", "from_default"}
    assert set(find_variables({"x": [closed], "y": 1})) == {"from_closure"}
    assert find_variables(42) == {}

    def outer(X):
        def inner(Y):
            return Y + MODULE_LEVEL
        return inner(X)

    assert set(find_variables(outer)) == {"from_globals"}  # globals referenced from nested code


def test_pipeline_op_registers_variable_inputs_and_skips_provided():
    from AFL.double_agent.PipelineOp import PipelineOp

    class Op(PipelineOp):
        PROVIDED_KEYS = ("model",)

        def __init__(self, score, input_variable=None, output_variable="out", name="Op"):
            super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        def calculate(self, dataset):
            return self

    comp, model = Variable("composition"), Variable("model")
    op = Op(score=lambda X: X + comp + model)
    assert op.input_variable == ["composition"]
    op = Op(score=lambda X: X + comp, input_variable="labels")
    assert op.input_variable == ["labels", "composition"]
    op = Op(score=lambda X: X)
    assert op.input_variable is None
