import numpy as np
import xarray as xr

from AFL.automation.APIServer.Client import Client
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify

from tiled.client import from_uri
from tiled.queries import Eq, Contains


class AutoSAS(PipelineOp):
    def __init__(
        self,
        sas_variable,
        sas_err_variable,
        resolution,
        output_prefix,
        q_dim,
        sample_dim,
        model_dim,
        server_id="localhost:5058",
        tiled_server_id="https://localhost:8000",
        tiled_api_key="key",
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
        self.tiled_server_id = tiled_server_id
        self.tiled_api_key = tiled_api_key
        self.fit_method = fit_method

        self.q_dim = q_dim
        self.sas_variable = sas_variable
        self.sas_err_variable = sas_err_variable

        self.sample_dim = sample_dim
        self.model_dim = model_dim
        self._banned_from_attrs.extend(["AutoSAS_client","tiled_client"])

    def construct_clients(self):
        """
        creates a client to talk to the AutoSAS server
        """

        self.AutoSAS_client = Client(
            self.server_id.split(":")[0], port=self.server_id.split(":")[1]
        )
        self.AutoSAS_client.login("AutoSAS_client")
        self.AutoSAS_client.debug(False)

        self.tiled_client = from_uri(
            self.tiled_server_id,
            api_key=self.tiled_api_key
        )

    def calculate(self, dataset):
        """
        Default class calculate runs a fit method on the input data and outputs a classification
        """
        self.construct_clients()

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

        all_chisq = report_json["all_chisq"]

        self.output[self._prefix_output("all_chisq")] = xr.DataArray(
            data = all_chisq, 
            dims = [self.sample_dim, self.model_dim],
            coords = [np.arange(len(all_chisq)), [m["name"] for m in list(report_json["model_fits"][0])]]
        )
        self.output[self._prefix_output("all_chisq")].attrs[
            "tiled_calc_id"
        ] = tiled_calc_id



        
        target = {}
        err = {}
        for idx,fit in enumerate(report_json["model_fits"]):
            for model in fit:
                target[model['name']] = {}
                err[model['name']] = {}
                
                for param in model["output_fit_params"]:
                    param_name = '_'.join([model['name'],param])
                    target[model['name']][param_name] = []
                    err[model['name']][param_name] = []
                    
        for idx,fit in enumerate(report_json["model_fits"]):
            for model in fit:
                for param in model["output_fit_params"]:
                    
                    param_name = '_'.join([model['name'],param])
                    
                    target[model['name']][param_name].append(model["output_fit_params"][param]["value"])
                    err[model['name']][param_name].append(model["output_fit_params"][param]["error"])


        
        # print("Writing out the data to output dictionary")
        #target it now a dictionary that allows for multiple things
        for key in list(target):
            
            model_dim = key+'_params'
            p_data = np.array([target[key][param] for param in list(target[key])])
            
            # print(p_data.shape)
            self.output[f"{key}_fit_val"] = xr.DataArray(
                data=p_data,
                coords={model_dim:list(target[key])},
                dims=[model_dim,self.sample_dim]
            )
            self.output[f"{key}_fit_val"].attrs[
                "tiled_calc_id"
            ] = tiled_calc_id

            e_data = np.array([err[key][param] for param in list(err[key])])
            
            self.output[f"{key}_fit_err"] = xr.DataArray(
                data=e_data,
                coords=[list(target[key]), np.arange(e_data.shape[1])],
                dims=[model_dim, self.sample_dim]
            )
            self.output[f"{key}_fit_err"].attrs[
                "tiled_calc_id"
            ] = tiled_calc_id

        result = self.tiled_client.search(Eq('uuid',tiled_calc_id))
        tiled_dict = {}
        for uuid,entry in list(result.items()):
            task = entry.metadata['array_name']
            # print(task)
            
            arr = entry[()]
            tiled_dict[task] = arr
            # print(arr)

        
        model_ids = list(target)
        for idx, mid in enumerate(model_ids):
            q_key = [i for i in list(tiled_dict) if ('fit_q' in i and mid in i)][0]

            
            for key in list(tiled_dict):
        
                if (mid in key) and ('fit_I' in key):
                    self.output[key] = xr.DataArray(
                        tiled_dict[key],
                        dims=[self.sample_dim,q_key],
                        coords=[np.arange(e_data.shape[1]), tiled_dict[q_key]]
                    )
                elif (mid in key) and ('residuals' in key):
                    self.output[key] = xr.DataArray(
                        tiled_dict[key],
                        dims=[self.sample_dim,q_key],
                        coords=[np.arange(e_data.shape[1]), tiled_dict[q_key]]
                    )
        
        return self

class ModelSelectBestChiSq(PipelineOp):
    def __init__(
        self,
        all_chisq_var,
        model_names_var,
        model_dim,
        sample_dim,
<<<<<<< HEAD
        chisq_variable,
        chisq_cutoff=1.0,
		complexity_order=None,
=======
        selection_method=None,
        output_prefix='BestChiSq',
>>>>>>> 1639035ec31457f2e9798396c79d35e465f90fc4
        name="ModelSelection",
        **kwargs
    ):
        
        output_variables = ["labels", "label_names"]
        super().__init__(
            name=name,
            input_variable=[all_chisq, ],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        
        self.sample_dim = sample_dim
        self.model_dim = model_dim
        self.model_names_var = model_names_var
        self.selection_method = selection_method
        
        self.all_chisq_var = all_chisq_var
        
        def calculate(self, dataset):        
            """Method for selecting the model based on the best chi-squared value"""
            self.output[self._output_prefix('labels')]

            labels = dataset[all_chisq_var].argmax(model_dim).values
            
            label_names = np.array([dataset[model_names_var][i].values for i in labels])
            
            self.output[self._prefix_output("labels")] = xr.DataArray(
                data=labels,
                dims=[self.sample_dim]
            )
            self.output[self._prefix_output("labels")].attrs[
                "tiled_calc_id"
            ] = tiled_calc_id

            self.output[self._prefix_output("label_names")] = xr.DataArray(
                data=label_nanes,
                dims=[self.sample_dim]
            )
            self.output[self._prefix_output("label_names")].attrs[
                "tiled_calc_id"
            ] = tiled_calc_id
            return self









