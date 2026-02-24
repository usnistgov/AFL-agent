"""
Unit tests for the AFL.double_agent.PipelineOp module.
"""

import pytest
import numpy as np
import xarray as xr
import json
import os

from tests.utils import MockPipelineOp
from AFL.double_agent import TreePipeline as tp
from AFL.double_agent import (Pipeline, LogLogTransform)
from sklearn.svm import SVC
from AFL.double_agent.data import (
    get_data_dir,
    list_datasets,
    load_dataset,
    example_dataset1,
)
from TreeHierarchy import (
    TreeHierarchy,
    json_decoder
)


@pytest.mark.unit
class TestClassificationPipeline:
    """Tests for the PipelineOp class."""
    def test_classifier_creation(self):
        data = load_dataset("example_classification_data")
        classification_def = json.loads(open(os.path.join(get_data_dir(), "example_tree_structure.json"), 'r').read())
        with Pipeline() as P:
           LogLogTransform("SAS_curves", "log_sas_curves")
           pipe = tp.ClassificationPipeline("SAS_curves", "predicted_labels", classification_def)
           assert isinstance(pipe, tp.ClassificationPipeline)
           assert isinstance(pipe.classifier, TreeHierarchy)
           assert isinstance(pipe.classifier.left, TreeHierarchy)
           assert isinstance(pipe.classifier.right, TreeHierarchy)
           assert isinstance(pipe.classifier.left.left, TreeHierarchy)
           assert isinstance(pipe.classifier.left.right, TreeHierarchy)
           assert isinstance(pipe.classifier.right.left, TreeHierarchy)
           assert isinstance(pipe.classifier.right.right, TreeHierarchy)
           assert isinstance(pipe.classifier.entity, SVC)
           assert isinstance(pipe.classifier.left.entity, SVC)
           assert isinstance(pipe.classifier.right.entity, SVC)

@pytest.mark.unit
class TestClassificationPipelineLoaded:
    """Tests for the PipelineOp class."""
    def test_classifier_load(self):
###        data = load_dataset("classification_data")
###        classification_def = json.loads(open(os.path.join(get_data_dir(), "classification_tree.json"), 'r').read())
###        pipe = tp.ClassificationPipeline("log_sas_curves", "predicted_labels", classification_def)
        save_path = os.path.join(get_data_dir(), "classification_pipeline.json")
        with Pipeline.read_json(str(save_path)) as P:
           assert isinstance(P[1], tp.ClassificationPipeline)
           assert isinstance(P[1].classifier, TreeHierarchy)
           assert isinstance(P[1].classifier.left, TreeHierarchy)
           assert isinstance(P[1].classifier.right, TreeHierarchy)
           assert isinstance(P[1].classifier.left.left, TreeHierarchy)
           assert isinstance(P[1].classifier.left.right, TreeHierarchy)
           assert isinstance(P[1].classifier.right.left, TreeHierarchy)
           assert isinstance(P[1].classifier.right.right, TreeHierarchy)
           assert isinstance(P[1].classifier.entity, SVC)
           assert isinstance(P[1].classifier.left.entity, SVC)
           assert isinstance(P[1].classifier.right.entity, SVC)

@pytest.mark.unit
class TestClassificationPipelinePerformance:
    """Tests for the PipelineOp class."""
    def test_classifier_load(self):
###        data = load_dataset("classification_data")
###        classification_def = json.loads(open(os.path.join(get_data_dir(), "classification_tree.json"), 'r').read())
###        pipe = tp.ClassificationPipeline("log_sas_curves", "predicted_labels", classification_def)
        save_path = os.path.join(get_data_dir(), "classification_pipeline.json")
        data = load_dataset("example_classification_data")
        ref = load_dataset("reference_predictions")
        with Pipeline.read_json(str(save_path)) as P:
            out = P.calculate(data)
            print(P[0].output_variable)
            np.testing.assert_array_equal(out["predicted_test_labels"].data, ref["reference_predictions"].data)




