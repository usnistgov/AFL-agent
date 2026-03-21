"""
Tests for TreePipeline PipelineOps.
"""

import numpy as np
import pytest
import xarray as xr

from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.TreePipeline import (
    ClassificationPipeline,
    IntEncoding,
    RegressionPipeline,
    ThresholdClassificationPipeline,
    TrimQ,
)


class DummyClassifier:
    def __init__(self):
        self.last_input = None

    def predict(self, X):
        self.last_input = X
        return np.array(["A"] * X.shape[0])


class DummyRegressor:
    def predict(self, X):
        return np.full(X.shape[0], 5.0)


@pytest.mark.unit
def test_classification_pipeline_output_and_pipeline_merge():
    ds = xr.Dataset({"features": (("sample", "f"), np.ones((3, 2)))})

    op = ClassificationPipeline(
        input_variable="features",
        output_variable="labels",
        model_definition=None,
    )
    op.set_classifier(DummyClassifier())

    result = op.calculate(ds)
    assert result is op
    assert "labels" in op.output
    assert op.output["labels"].dims == ("sample",)
    assert np.all(op.output["labels"].values == "A")

    pipeline = Pipeline(ops=[op])
    merged = pipeline.calculate(ds)
    assert "labels" in merged


@pytest.mark.unit
def test_classification_pipeline_custom_sample_dim_and_axis_order():
    ds = xr.Dataset(
        {
            "features": (
                ("feature", "item"),
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [10.0, 20.0, 30.0],
                    ]
                ),
            )
        },
        coords={"item": ["s1", "s2", "s3"]},
    )

    classifier = DummyClassifier()
    op = ClassificationPipeline(
        input_variable="features",
        output_variable="labels",
        model_definition=None,
        sample_dim="item",
    )
    op.set_classifier(classifier)

    op.calculate(ds)

    np.testing.assert_array_equal(
        classifier.last_input,
        np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        ),
    )
    assert op.output["labels"].dims == ("item",)
    np.testing.assert_array_equal(op.output["labels"].coords["item"].values, ds.coords["item"].values)


@pytest.mark.unit
def test_classification_pipeline_legacy_axis_zero_sample_fallback():
    ds = xr.Dataset(
        {
            "features": (
                ("test_sample", "f"),
                np.array(
                    [
                        [1.0, 10.0],
                        [2.0, 20.0],
                        [3.0, 30.0],
                    ]
                ),
            )
        },
        coords={"test_sample": ["s1", "s2", "s3"]},
    )

    op = ClassificationPipeline(
        input_variable="features",
        output_variable="labels",
        model_definition=None,
    )
    op.set_classifier(DummyClassifier())

    op.calculate(ds)

    out = op.output["labels"]
    assert out.dims == ("test_sample",)
    np.testing.assert_array_equal(out.coords["test_sample"].values, ds.coords["test_sample"].values)


@pytest.mark.unit
def test_regression_pipeline_selective_update():
    ds = xr.Dataset(
        {
            "features": (("sample", "f"), np.ones((3, 2))),
            "phase": ("sample", np.array(["A", "B", "A"])),
            "param": ("sample", np.array([1.0, 1.0, 1.0])),
        }
    )

    op = RegressionPipeline(
        input_variable="features",
        output_variable="param",
        key_variable="phase",
        morphology="A",
        model_definition=None,
    )
    op.set_regressor(DummyRegressor())

    op.calculate(ds)
    out = op.output["param"].values
    assert out[0] == 5.0
    assert out[1] == 1.0
    assert out[2] == 5.0


@pytest.mark.unit
def test_regression_pipeline_custom_sample_dim_and_axis_order():
    ds = xr.Dataset(
        {
            "features": (
                ("f", "row"),
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [10.0, 20.0, 30.0],
                    ]
                ),
            ),
            "phase": ("row", np.array(["A", "B", "A"])),
            "param": ("row", np.array([1.0, 1.0, 1.0])),
        },
        coords={"row": ["r1", "r2", "r3"]},
    )

    op = RegressionPipeline(
        input_variable="features",
        output_variable="param",
        key_variable="phase",
        morphology="A",
        model_definition=None,
        sample_dim="row",
    )
    op.set_regressor(DummyRegressor())

    op.calculate(ds)
    out = op.output["param"]
    np.testing.assert_array_equal(out.values, np.array([5.0, 1.0, 5.0]))
    assert out.dims == ("row",)
    np.testing.assert_array_equal(out.coords["row"].values, ds.coords["row"].values)


@pytest.mark.unit
def test_threshold_classification_pipeline():
    ds = xr.Dataset(
        {
            "labels": ("sample", np.array(["mix", "mix"], dtype=object)),
            "a": ("sample", np.array([0.8, 0.5])),
            "b": ("sample", np.array([0.2, 0.5])),
        }
    )
    components = {"mix": ["a", "b"]}

    op = ThresholdClassificationPipeline(
        input_variable="labels",
        output_variable="out_labels",
        components=components,
        threshold=0.7,
    )
    op.calculate(ds)
    labels = op.output["out_labels"].values
    assert labels[0] == "a"
    assert labels[1] == "mix"


@pytest.mark.unit
def test_threshold_classification_pipeline_custom_sample_dim():
    ds = xr.Dataset(
        {
            "labels": ("batch", np.array(["mix", "mix"], dtype=object)),
            "a": ("batch", np.array([0.8, 0.5])),
            "b": ("batch", np.array([0.2, 0.5])),
        },
        coords={"batch": ["b1", "b2"]},
    )
    components = {"mix": ["a", "b"]}

    op = ThresholdClassificationPipeline(
        input_variable="labels",
        output_variable="out_labels",
        components=components,
        threshold=0.7,
        sample_dim="batch",
    )
    op.calculate(ds)

    out = op.output["out_labels"]
    np.testing.assert_array_equal(out.values, np.array(["a", "mix"], dtype=object))
    assert out.dims == ("batch",)
    np.testing.assert_array_equal(out.coords["batch"].values, ds.coords["batch"].values)


@pytest.mark.unit
def test_int_encoding_custom_sample_dim():
    ds = xr.Dataset(
        {"labels": ("batch", np.array(["A", "B", "A"], dtype=object))},
        coords={"batch": ["b1", "b2", "b3"]},
    )

    op = IntEncoding(
        input_variable="labels",
        output_variable="encoded",
        classes=["A", "B"],
        sample_dim="batch",
    )
    op.calculate(ds)

    out = op.output["encoded"]
    np.testing.assert_array_equal(out.values, np.array([0, 1, 0]))
    assert out.dims == ("batch",)
    np.testing.assert_array_equal(out.coords["batch"].values, ds.coords["batch"].values)


@pytest.mark.unit
def test_trimq_does_not_mutate_input_coords():
    q = np.array([0.0, 1.0, 2.0])
    data = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 3.0, 4.0],
        ]
    )
    ds = xr.Dataset({"I": (("sample", "q"), data)}, coords={"q": q})

    new_q = [0.0, 0.5, 1.0, 1.5, 2.0]
    op = TrimQ(
        input_variable="I",
        output_variable="I_trim",
        input_index="q",
        output_index="q_trim",
        new_values=new_q,
    )
    op.calculate(ds)

    assert np.array_equal(ds.coords["q"].values, q)
    out = op.output["I_trim"]
    assert out.dims == ("sample", "q_trim")
    assert np.array_equal(out.coords["q_trim"].values, np.array(new_q))
