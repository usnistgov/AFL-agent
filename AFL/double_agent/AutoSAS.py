"""
PairMetrics are PipelineOps that produce pair matrices as results

"""

import numpy as np
import xarray as xr

from AFL.automation.APIServer.Client import Client
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify


class AutoSAS(PipelineOp):
    def __init__(
        self,
        sas_variable,
        sas_err_variable,
        resolution,
        output_prefix,
        q_dim,
        sample_dim,
        server_id="localhost:5058",
        tiled_id="localhost:8000",
        fit_method=None,
        name="AutoSAS_fit",
    ):
        output_variables = ["all_chisq"]
        super().__init__(
            name=name,
            input_variable=[q_dim, sas_variable, sas_err_variable],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        self.server_id = server_id
        self.fit_method = fit_method

        self.q_dim = q_dim
        self.sas_variable = sas_variable
        self.sas_err_variable = sas_err_variable

        self.sample_dim = sample_dim
        self._banned_from_attrs.extend(["AutoSAS_client"])

    def construct_client(self):
        """
        creates a client to talk to the AutoSAS server
        """

        self.AutoSAS_client = Client(
            self.server_id.split(":")[0], port=self.server_id.split(":")[1]
        )
        self.AutoSAS_client.login("AutoSAS_client")
        self.AutoSAS_client.debug(False)

    def calculate(self, dataset):
        """
        Default class calculate runs a fit method on the input data and outputs a classification
        """
        self.construct_client()

        sub_dataset = dataset[[self.q_dim, self.sas_variable, self.sas_err_variable]]
        db_uuid = self.AutoSAS_client.deposit_obj(obj=sub_dataset)

        self.AutoSAS_client.enqueue(
            task_name="set_sasdata",
            db_uuid=db_uuid,
            sample_dim=self.sample_dim,
            q_variable=self.q_dim,
            sas_variable=self.sas_variable,
            sas_err_variable=self.sas_err_variable,
        )

        tiled_calc_id = self.AutoSAS_client.enqueue(
            task_name="fit_models", fit_method=self.fit_method
        )

        report_json = self.AutoSAS_client.enqueue(
            task_name="build_report", interactive=True
        )["return_val"]

        all_chisq = report_json["best_fits"]["all_chisq"]

        self.output[self._prefix_output("all_chisq")] = xr.DataArray(
            best_chisq, dims=[self.sample_dim]
        )
        self.output[self._prefix_output("all_chisq")].attrs[
            "tiled_calc_id"
        ] = tiled_calc_id


        target = {}
        err = {}
        for param in self.target_fit_params:
            target[param] = []
            err[param] = []

        for idx,fit in enumerate(report_json["model_fits"]):
            for model in fit:
                for param in model["output_fit_params"]:
                    param_name = '_'.join([model['name'],param])
                    
                    target[param_name].append(model["output_fit_params"][param]["value"])
                    err[param_name].append(model["output_fit_params"][param]["error"])
        
        #if target == False:
        #    raise ValueError(f'The target model {self.target_model} is not in the config')

        
        #target it now a dictionary that allows for multiple things
        for key in list(target):
            self.output[self._prefix_output(f"{key}_fit_val")] = xr.DataArray(
                target[key], dims=[self.sample_dim]
            )
            self.output[self._prefix_output(f"{key}_fit_val")].attrs[
                "tiled_calc_id"
            ] = tiled_calc_id

            
            self.output[self._prefix_output(f"{key}_fit_err")] = xr.DataArray(
                err[key], dims=[self.sample_dim]
            )
            self.output[self._prefix_output(f"{key}_fit_err")].attrs[
                "tiled_calc_id"
            ] = tiled_calc_id




        #### TO DO ####
        # - validate the current changes to the pipeline op
        # - extract the intensities and q-vectors from tiled and store them here
        return self

class ModelSelect_parsimony(PipelineOP):
    def __init__(
        self,
        output_prefix,
        sample_dim,
        chisq_variable,
        chisq_cutoff=1.0,
        server_id="localhost:5058",
        tiled_id="localhost:8000",
        name="ModelSelection",
    ):
        output_variables = ["labels" , "best_chisq", "label_names"]
        super().__init__(
            name=name,
            input_variable=[q_dim, sas_variable, sas_err_variable],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        self.server_id = server_id

        self.sample_dim = sample_dim
        self.chisq_variable = chisq_variable
        self.chisq_cutoff = chisq_cutoff

        #self._banned_from_attrs.extend(["AutoSAS_client"])

        def calculate(self, dataset):
            """ Apply this `PipelineOp` to the supplied `xarray.dataset`"""
            all_chisq = dataset[self.chisq_variable]
            
            self.output[self._output_prefix('labels')]
            return self