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

from AFL.double_agent.SAS_model_fit_driver import SAS_model_fit, sas_wrapper
import sasmodels


class SASfit_classifier(PipelineOp):
    def __init__(self, sas_variable, sas_err_variable, input_models, resolution, output_prefix, q_dim, sample_dim, server_id='localhost:5058', fit_method=None, name="SASfit_classifier"):
        output_variables = ['labels','label_names','best_chisq','best_chisq']
        super().__init__(
            name=name,
            input_variable = [q_dim, sas_variable, sas_err_variable],
            output_variable = [output_prefix+'_'+o for o in listify(output_variables)],
            output_prefix=output_prefix
        )
        self.server_id = server_id
        self.fit_method = fit_method
        
        self.q_dim = q_dim
        self.sas_variable = sas_variable
        self.sas_err_variable = sas_err_variable
        
        
        self.input_models = input_models
        self.sample_dim = sample_dim
        self.labels=None
        
        self._banned_from_attrs.extend(['SASfit_client','input_models'])
        
    def construct_client(self, dataset):
        """
        creates a client to talk to the SASfit server
        """
        q = dataset[self.q_dim].values
        
        self.SASfit_client = Client(
            self.server_id.split(':')[0],
            port=self.server_id.split(':')[1]
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
        self.construct_client(dataset)
        
        q = dataset[self.q_dim].values.tolist()
        Is = dataset[self.sas_variable].values.tolist()
        dIs = dataset[self.sas_err_variable].values.tolist()
        
        tiled_calc_id = self.SASfit_client.enqueue(
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
        self.output[self._prefix_output("labels")].attrs['tiled_calc_id'] = tiled_calc_id
        self.output[self._prefix_output("label_names")] = xr.DataArray(label_names, dims=[self.sample_dim])
        self.output[self._prefix_output("label_names")].attrs['tiled_calc_id'] = tiled_calc_id
        self.output[self._prefix_output("best_chisq")] = xr.DataArray(best_chisq, dims=[self.sample_dim])
        self.output[self._prefix_output("best_chisq")].attrs['tiled_calc_id'] = tiled_calc_id
        return self



################
# trying without using the base class. Should talk to Tyler about the methodology. Following module is the same as above but it has just a few more outputs in the calculate method 
################
class SASfit_fit_extract(PipelineOp):
    def __init__(self, sas_variable, sas_err_variable, input_models, resolution, output_prefix, q_dim, sample_dim, target_model, target_fit_params, server_id='localhost:5058', fit_method=None, name='SASfit_fit_extract'):
        output_variables = ['labels','label_names','best_chisq','best_chisq','fit_values','fit_err_values']
        super().__init__(
            name=name,
            input_variable = [q_dim, sas_variable, sas_err_variable],
            output_variable = [output_prefix+'_'+o for o in listify(output_variables)],
            output_prefix = output_prefix
        )
        self.server_id = server_id

        #necessary to do the sas_fitting. These data variables/coordinates will build the SAS1D object and the target model and parameters to extract
        self.q_dim = q_dim
        self.sas_variable = sas_variable
        self.sas_err_variable = sas_err_variable
        
        self.target_model = target_model
        self.target_fit_params = target_fit_params
        
        self.input_models = input_models
        if self.target_model not in [model['name'] for model in self.input_models]:
            raise ValueError("Hey, the target model is not in the supplied input models")
        
        self.sample_dim = sample_dim
        self.fit_method = fit_method
        
        self._banned_from_attrs.extend(['SASfit_client','input_models'])

        
    def construct_client(self, dataset):
        """
        creates a client to talk to the SASfit server
        """
        q = dataset[self.q_dim].values
        
        self.SASfit_client = Client(
            self.server_id.split(':')[0],
            port=self.server_id.split(':')[1]
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
        self.construct_client(dataset)
        
        q = dataset[self.q_dim].values.tolist()
        Is = dataset[self.sas_variable].values.tolist()
        dIs = dataset[self.sas_err_variable].values.tolist()
        
        tiled_calc_id = self.SASfit_client.enqueue(
            task_name="fit_models",
            q = q,
            I = Is,
            dI = dIs,
            fit_method = self.fit_method
        )


        #calls the report rom the fit that was just run
        report_json = self.SASfit_client.enqueue(
            task_name="build_report",
            interactive = True
        )['return_val']

        #extract the labels and fit results 
        labels = report_json['best_fits']['model_idx']
        best_chisq = report_json['best_fits']['lowest_chisq']
        label_names = report_json['best_fits']['model_name']

        target = []
        err = []
        for idx, model in enumerate(report_json['best_fits']['model_params']):
            if model['name'] == self.target_model:
                target.append(model['output_fit_params'][self.target_fit_params]['value'])
                err.append(model['output_fit_params'][self.target_fit_params]['error'])
            else:
                #we may want something different than np.nan
                target.append(np.nan)
                err.append(np.nan)
        
        #mandatory for each pipeline operation
        self.output[self._prefix_output("labels")] = xr.DataArray(labels, dims=[self.sample_dim])
        self.output[self._prefix_output('labels')].attrs['tiled_calc_id'] = tiled_calc_id
        self.output[self._prefix_output("label_names")] = xr.DataArray(label_names, dims=[self.sample_dim])
        self.output[self._prefix_output('label_names')].attrs['tiled_calc_id'] = tiled_calc_id
        self.output[self._prefix_output("best_chisq")] = xr.DataArray(best_chisq, dims=[self.sample_dim])
        self.output[self._prefix_output('best_chisq')].attrs['tiled_calc_id'] = tiled_calc_id

        
        self.output[self._prefix_output("fit_values")] = xr.DataArray(target, dims=[self.sample_dim])
        self.output[self._prefix_output("fit_err_values")] = xr.DataArray(err, dims=[self.sample_dim])

        for output_var in ['fit_values','fit_err_values']:
            self.output[self._prefix_output(output_var)].attrs['target_model'] = self.target_model
            self.output[self._prefix_output(output_var)].attrs['target_fit_params'] = self.target_fit_params
            self.output[self._prefix_output(output_var)].attrs['tiled_calc_id'] = tiled_calc_id

        return self

        









