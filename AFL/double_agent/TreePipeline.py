"""
PipelineOps for tree-based classification/regression and simple post-processing.

These ops integrate with the external TreeHierarchy package.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import xarray as xr
from typing_extensions import Self

from AFL.double_agent.PipelineOp import PipelineOp
from TreeHierarchy import json_decoder

#PipelineOp constructor for classification tree
#The tree itself is defined in TreeHierarchy
#This constructor follows the expected PipelineOp syntax
#   input_variable:  the name of the input feature in the xarray
#   output_variable:  the name of the variable to add/modify in the xarray dataset
#   model_definition:  A dictionary containing an encoding of a TreeHierarchy object. The encoder is contained in treeHierarchy.
class ClassificationPipeline(PipelineOp):
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        model_definition: Optional[dict],
        name: str = "Classifier",
        sample_dim: str = "sample",
    ):
        super().__init__(
            input_variable=input_variable,
            output_variable=output_variable,
            name=name,
        )
        self.sample_dim = sample_dim
        self.classifier = json_decoder(model_definition) if model_definition else None
        self._banned_from_attrs.extend(["classifier"])

    def set_classifier(self, classifier_instance):
        self.classifier = classifier_instance

    def calculate(self, dataset: xr.Dataset) -> Self:
        if self.classifier is None:
            raise ValueError("Classifier is not set. Provide model_definition or call set_classifier().")

        data = self._get_variable(dataset)
        predicted_classes = self.classifier.predict(data.values)
        self.output[self.output_variable] = xr.DataArray(
            predicted_classes,
            dims=[self.sample_dim],
        )
        return self

#PipelineOp constructor for a regressor
#This constructor follows the expected PipelineOp syntax, with some important considerations
#   input_variable:  the name of the input feature in the xarray
#   output_variable:  the name of the variable to add/modify in the xarray dataset
#   key_variable: the name of the variable that contains morphology information in the xarray, could be ground_truth_labels, predicted_labels, etc.
#   morphology: the morphology that this model is trained on
#   model_Efinition: a dictionary containing a complete definition of a trained classification model, the encoder in TreeHierarchy also works for this
#NOTE: Each regressor only works for one parameter for one morphology, if multiple morphologies share a parameter i.e., radius is common to many morphologies, then they shuold each operate on the SAME output_variable. 
#Each RegressionPipeline will only modify output_variable where key_variable==morphology, place mulptiple PipelineOps in the same pipeline to perform regression over all parameters and morphologies
class RegressionPipeline(PipelineOp):
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        key_variable: str,
        morphology: str,
        model_definition: Optional[dict],
        name: str = "Regressor",
        sample_dim: str = "sample",
    ):
        super().__init__(
            input_variable=input_variable,
            output_variable=output_variable,
            name=name,
        )
        self.key_variable = key_variable
        self.morphology = morphology
        self.sample_dim = sample_dim
        self.regression = json_decoder(model_definition) if model_definition else None
        self._banned_from_attrs.extend(["regression"])

    def set_regressor(self, regressor_instance):
        self.regression = regressor_instance

    def calculate(self, dataset: xr.Dataset) -> Self:
        if self.regression is None:
            raise ValueError("Regressor is not set. Provide model_definition or call set_regressor().")

        data = self._get_variable(dataset)
        key = dataset[self.key_variable]
        if self.sample_dim not in key.dims:
            raise ValueError(
                f"key_variable '{self.key_variable}' must contain dim '{self.sample_dim}'."
            )
        key_values = key.values
        inds = np.where(np.equal(key_values, self.morphology))[0]

        if self.output_variable in dataset.data_vars:
            output_da = dataset[self.output_variable].copy()
            output = output_da.values
        else:
            output = np.full(key_values.shape[0], np.nan)
            coords = {}
            if self.sample_dim in key.coords:
                coords[self.sample_dim] = key.coords[self.sample_dim]
            output_da = xr.DataArray(output, dims=[self.sample_dim], coords=coords)

        if len(inds) > 0:
            predictions = self.regression.predict(data.values[inds])
            output[inds] = predictions.reshape(-1)

        output_da.values = output
        self.output[self.output_variable] = output_da
        return self

class ThresholdClassificationPipeline(PipelineOp):
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        components: Dict[str, List[str]],
        threshold: float,
        name: str = "mixture_separation",
        sample_dim: str = "sample",
    ):
        super().__init__(
            input_variable=input_variable,
            output_variable=output_variable,
            name=name,
        )
        self.components = components
        self.threshold = threshold
        self.sample_dim = sample_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        data = self._get_variable(dataset)
        data_values = data.values
        labs = []
        for i in range(data_values.shape[0]):
            d = data_values[i]
            comps = self.components[d]
            measures = np.array([dataset[c].values[i] for c in comps])
            total = np.sum(measures)
            if total == 0:
                labs.append(d)
                continue
            portions = measures / total
            if np.any(portions >= self.threshold):
                labs.append(comps[np.where(portions >= self.threshold)[0][0]])
            else:
                labs.append(d)
        self.output[self.output_variable] = xr.DataArray(labs, dims=[self.sample_dim])
        return self

class FlatAddition(PipelineOp):
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        value: float,
        name: str = "flat_addition",
    ):
        super().__init__(
            input_variable=input_variable,
            output_variable=output_variable,
            name=name,
        )
        self.value = value

    def calculate(self, dataset: xr.Dataset) -> Self:
        data = self._get_variable(dataset)
        self.output[self.output_variable] = data + self.value
        return self

class IntEncoding(PipelineOp):
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        classes: Iterable[str],
        name: str = "label_encoder",
        sample_dim: str = "sample",
    ):
        super().__init__(
            input_variable=input_variable,
            output_variable=output_variable,
            name=name,
        )
        self.classes = classes
        self.encoding = {c:i for i,c in enumerate(classes)}
        self.sample_dim = sample_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        data = self._get_variable(dataset)
        values = []
        for label in data.values:
            if label not in self.encoding:
                raise ValueError(f"Label '{label}' not in encoding classes.")
            values.append(self.encoding[label])
        self.output[self.output_variable] = xr.DataArray(values, dims=[self.sample_dim])
        return self

class TrimQ(PipelineOp):
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        input_index: str,
        output_index: str,
        new_values: Iterable[float],
        name: str = "q interpolation",
    ):
        super().__init__(
            input_variable=input_variable,
            output_variable=output_variable,
            name=name,
        )
        self.input_index = input_index
        self.output_index = output_index
        self.new_values = list(new_values)

    def calculate(self, dataset: xr.Dataset) -> Self:
        data = self._get_variable(dataset)
        old_q = dataset[self.input_index].values
        old_I = data.values
        new_values = np.asarray(self.new_values)
        new_I = np.array(
            [np.interp(new_values, old_q, old_I[i, :]) for i in range(old_I.shape[0])]
        )
        new_dims = [
            dim if dim != self.input_index else self.output_index
            for dim in data.dims
        ]
        new_coords = {
            dim: data.coords[dim] for dim in data.dims if dim in data.coords and dim != self.input_index
        }
        new_coords[self.output_index] = new_values
        self.output[self.output_variable] = xr.DataArray(
            new_I,
            dims=new_dims,
            coords=new_coords,
        )
        return self
