from AFL.double_agent import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
#import tune_all_decisions as tad
import itertools
import joblib
from io import BytesIO
import xarray as xr
import json
import AFL.double_agent.TreeHierarchy as te
from sklearn.metrics import classification_report as cr
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

#PipelineOp constructor for classification tree
#The tree itself is defined in TreeHierarchy
#This constructor follows the expected PipelineOp syntax
#   input_variable:  the name of the input feature in the xarray
#   output_variable:  the name of the variable to add/modify in the xarray dataset
#   model_definition:  A dictionary containing an encoding of a TreeHierarchy object. The encoder is contained in treeHierarchy.
class ClassificationPipeline(PipelineOp):
    def __init__(self, input_variable, output_variable, model_definition, name="Classifier"):
        super().__init__(
                input_variable=input_variable,
                output_variable=output_variable,
                name=name
        )
        self.classifier = te.json_decoder(model_definition)

    def set_classifier(self, classifier_instance):
        self.classifier = classifier_instance

    def calculate(self, dataset):
        data = self._get_variable(dataset)
        predicted_classes = self.classifier.predict(data)
        dataset[self.output_variable] = ('sample', predicted_classes)
        return(self)

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
    def __init__(self, input_variable, output_variable, key_variable, morphology, model_definition, name="Classifier"):
        super().__init__(
                input_variable=input_variable,
                output_variable=output_variable,
                name=name
        )
        self.key_variable = key_variable
        self.morphology = morphology
        self.regression = te.json_decoder(model_definition)

    def calculate(self, dataset):
        data = self._get_variable(dataset)
        key = dataset[self.key_variable].data
        print(np.unique(key))
        print(self.morphology)
        inds = np.where(np.equal(key, self.morphology))[0]
        predictions = self.regression.predict(data[inds])
        if self.output_variable in dataset.data_vars:
            output = dataset[self.output_variable].data
        else:
            output = np.nan * np.ones(data.shape[0])
        print("INDS")
        print(inds.shape)
        print("PREDS")
        print(predictions.shape)
        output[inds] = predictions.reshape(-1)
        dataset[self.output_variable] = ('sample', output)
        return(self)

class ThresholdClassificationPipeline(PipelineOp):
    def __init__(self, input_variable, output_variable, components, threshold, name = "mixture_separation"):
        super().__init__(input_variable = input_variable,
                         output_variable = output_variable,
                         name = name)
        self.components = components
        self.threshold = threshold

    def calculate(self, dataset):
        data = self._get_variable(dataset)
        labs = []
        for i in range(data.shape[0]):
            d = data.data[i]
            print(d)
            print(type(d))
            comps = self.components[d]
            measures = np.array([dataset[c].data[i] for c in comps])
            portions = measures/np.sum(measures)
            print(np.where(portions > self.threshold)[0])
            if any(portions >= self.threshold):
                labs += [comps[np.where(portions >= self.threshold)[0][0]]]
            else:
                labs += [d]
        dataset[self.output_variable] = ('sample', labs)
        return(self)

class FlatAddition(PipelineOp):
    def __init__(self, input_variable, output_variable, value, name = "flat_addition"):
        super().__init__(input_variable = input_variable,
                         output_variable = output_variable,
                         name = name)
        self.value = value

    def calculate(self, dataset):
        data = self._get_variable(dataset)
        dataset[self.output_variable] = data+self.value
        return(self)

class IntEncoding(PipelineOp):
    def __init__(self, input_variable, output_variable, classes, name = "label_encoder"):
        super().__init__(input_variable = input_variable,
                         output_variable = output_variable,
                         name = name)
        self.classes = classes
        self.encoding = {c:i for i,c in enumerate(classes)}

    def calculate(self, dataset):
        data = self._get_variable(dataset)
        dataset[self.output_variable] = ('sample', [self.encoding[l] for l in data.data])
        return(self)
