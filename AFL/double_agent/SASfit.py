"""
PairMetrics are PipelineOps that produce pair matrices as results

"""
import xarray as xr
import numpy as np

import sklearn.gaussian_process
import sklearn.gaussian_process.kernels

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify

import seaborn as sns
from AFL.automation.APIServer.Client import Client

from SAS_model_fit_driver import SAS_model_fit, sas_wrapper
import sasmodels


class SASfit_classifier(PipelineOp):
    def __init__(self, sas_variable, sas_err_variable, output_variables, input_models, resolution, output_prefix, q_dim, sample_dim, server_id='http://localhost:5058', fit_method=None, name="SASfit"):
        super().__init__(
            name=name,
            input_variable = [q_dim, sas_variable, sas_err_variable],
            output_variable = [output_prefix+'_'+o for o in listify(output_variables)]
        )
        self.server_id = server_id
        
        self.q_dim = q_dim
        self.sas_variable = sas_variable
        self.sas_err_variable = sas_err_variable
        
        
        self.input_models = input_models
        self.sample_dim = sample_dim
        self.labels=None
        self.fit_method = fit_method
        self.fit_tiled_id = None
        
        self._banned_from_attrs.append('SASfit_client','input_models')
        
    def construct_client(self, dataset):
        """
        creates a client to talk to the SASfit server
        """
        q = dataset[self.q_dim].values
        
        self.SASfit_client = Client(
            self.server_id.spit(':')[0],
            port=self.server_id.spit(':')[1]
        )
        self.SASfit_client.login('SASfit_Client')
        self.SASfit_client.debug(False)
        
        self.SASfit_client.set_config(
            q_range=(min(q), max(q)),
            model_inputs = self.input_models
        )
        
    def calculate(self, dataset):
        """
        Default class calculate runs a fit method on the input data and outputs a classification
        """
        q = dataset[self.q_dim].values.tolist()
        Is = dataset[self.sas_variable].values.tolist()
        dIs = dataset[self.sas_err_variable].values.tolist()
        
        self.SASfit_client.enqueue(
            task_name="fit_models",
            q = q,
            I = Is,
            dI = dIs,
            fit_method = self.fit_method
        )
        
        report_json = self.SASfit_client.enqueue(
            task_name="build_report",
            interactive = True
        )['return_val']
        
        labels = report_json['best_fits']['model_idx']
        best_chisq = report_json['best_fits']['lowest_chisq']
        label_names = report_json['best_fits']['model_name']
        
        #mandatory for each pipeline operation
        self.output[self._prefix_output("labels")] = xr.DataArray(labels, dims=[self.sample_dim])
        self.output[self._prefix_output("label_names")] = xr.DataArray(label_names, dims=[self.sample_dim])
        self.output[self._prefix_output("best_chisq")] = xr.DataArray(best_chisq, dims=[self.sample_dim])
        return self


class SASfit_fit_extract(SASfit):
    def __init__(self, input_variable, output_variable, dim, params=None, name='SASLabeler'):
        super().__init__(
            name=name,
            input_variable=input_variable,
            output_variable=output_variable,
            dim=dim,
            params=params,
            q_SAS
        )

    def calculate():
        pass
        
        return self
        









