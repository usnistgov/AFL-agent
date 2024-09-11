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
        #model_dim,
        sample_dim,
        output_prefix='BestChiSq',
        name="ModelSelection_BestChiSq",
    ):
        
        output_variables = ["labels", "label_names"]
        super().__init__(
            name=name,
            input_variable=[all_chisq_var, model_names_var],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        
        self.sample_dim = sample_dim
        #self.model_dim = model_dim
        self.model_names_var = model_names_var
        
        self.all_chisq_var = all_chisq_var

    def calculate(self, dataset):        
        """Method for selecting the model based on the best chi-squared value"""
        
        self.dataset = dataset.copy(deep=True)
        
        labels = self.dataset[self.all_chisq_var].argmin(self.model_names_var).values
        label_names = np.array([self.dataset[self.model_names_var][i].values for i in labels])
        bestChiSq = self.dataset[self.all_chisq_var].min(self.model_names_var).values

        self.output[self._prefix_output("labels")] = xr.DataArray(
            data=labels,
            dims=[self.sample_dim]
        )
        
        self.output[self._prefix_output("ChiSq")] = xr.DataArray(
            data=bestChiSq,
            dims=[self.sample_dim]
        )

        self.output[self._prefix_output("label_names")] = xr.DataArray(
            data=label_names,
            dims=[self.sample_dim]
        )
        return self


class ModelSelectParsimony(PipelineOp):
    def __init__(
        self,
        all_chisq_var,
        model_names_var,
        sample_dim,
        cutoff_threshold=1.0,
        model_complexity=None,
        server_id="localhost:5058",
        output_prefix='Parsimony',
        name="ModelSelection_Parsimony",
        **kwargs
    ):
        
        output_variables = ["labels", "label_names"]
        super().__init__(
            name=name,
            input_variable=[all_chisq_var, model_names_var],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        
        self.sample_dim = sample_dim
        self.model_names_var = model_names_var
        self.cutoff_threshold = cutoff_threshold 
        self.model_complexity = model_complexity
        self.all_chisq_var = all_chisq_var
        self.server_id = server_id
    
    def construct_clients(self):
        """
        creates a client to talk to the AutoSAS server
        """

        self.AutoSAS_client = Client(
            self.server_id.split(":")[0], port=self.server_id.split(":")[1]
        )
        self.AutoSAS_client.login("AutoSAS_client")
        self.AutoSAS_client.debug(False)

    def calculate(self, dataset):        
        """Method for selecting the model based on parsimony given a user defined ChiSq threshold """
        
        self.construct_clients()
        
        self.dataset = dataset.copy(deep=True)

        bestChiSq_labels = self.dataset[self.all_chisq_var].argmin(self.model_names_var)
        bestChiSq_label_names = np.array([self.dataset[self.model_names_var][i].values for i in bestChiSq_labels.values])
        
        ### default behavior is that complexity is determined by number of free parameters. 
        ### this is an issue if the number of parameters is the same between models. You bank on them having wildly different ChiSq vals
        ### could use a neighbor approach or some more intelligent selection methods
        if self.model_complexity is None:
            print('aggregating complexity')
            aSAS_config = self.AutoSAS_client.get_config('all',interactive=True)['return_val']
            order = []
            for model in aSAS_config['model_inputs']:
            #for model in dataset[self.model_names_var].values:
                n_params = 0
                for p in model['fit_params']:
                    if model['fit_params'][p]['bounds'] != None:
                        n_params +=1
                order.append(n_params)
            print(order)
            print(np.argsort(order))
            self.model_complexity = np.argsort(order).tolist()


       #as written in dev full of jank...
        replacement_labels = bestChiSq_labels.copy(deep=True)
        all_chisq = self.dataset[self.all_chisq_var]
        sorted_chisq = all_chisq.sortby(self.model_names_var, ascending=False).values

        min_diff_chisq =  np.array([row[1] - row[0] for row in sorted_chisq])
        next_best_idx = np.array([np.argpartition(row,1)[1] for row in all_chisq])

        for idx in range(len(replacement_labels)):
            chisq_set = all_chisq.min(dim=self.model_names_var).values

            if (min_diff_chisq[idx] <= self.cutoff_threshold):
                best_model_index = replacement_labels[idx]
                next_best_index = next_best_idx[idx]
                bm_rank = self.model_complexity.index(best_model_index)
                nbm_rank = self.model_complexity.index(next_best_index)
                
                if (bm_rank > nbm_rank):
                    replacement_labels[idx] = next_best_index

        

        self.output[self._prefix_output("labels")] = xr.DataArray(
            data=replacement_labels,
            dims=[self.sample_dim]
        )
        
        self.output[self._prefix_output("label_names")] = xr.DataArray(
            data=[self.dataset[self.model_names_var].values[i] for i in replacement_labels],
            dims=[self.sample_dim]
        )
        return self


class ModelSelectAIC(PipelineOp):
    def __init__(
        self,
        all_chisq_var,
        model_names_var,
        sample_dim,
        output_prefix='AIC',
        name="ModelSelectionAIC",
        server_id="localhost:5058",
        **kwargs
    ):
        
        output_variables = ["labels", "label_names", "AIC"]
        super().__init__(
            name=name,
            input_variable=[all_chisq_var, model_names_var],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
       
        self.server_id = server_id
        self.sample_dim = sample_dim
        self.model_names_var = model_names_var
        self.all_chisq_var = all_chisq_var
    
    def construct_clients(self):
        """
        creates a client to talk to the AutoSAS server
        """

        self.AutoSAS_client = Client(
            self.server_id.split(":")[0], port=self.server_id.split(":")[1]
        )
        self.AutoSAS_client.login("AutoSAS_client")
        self.AutoSAS_client.debug(False)

    def calculate(self, dataset):        
        """Method for selecting the model based on parsimony given a user defined ChiSq threshold """
        
        self.construct_clients()
        self.dataset = dataset.copy(deep=True)

        bestChiSq_labels = self.dataset[self.all_chisq_var].argmin(self.model_names_var).values
        bestChiSq_label_names = np.array([self.dataset[self.model_names_var][i].values for i in bestChiSq_labels])
        
        aSAS_config = self.AutoSAS_client.get_config('all',interactive=True)['return_val']
        n = []
        for model in aSAS_config['model_inputs']:
        #for model in dataset[self.model_names_var].values:
            n_params = 0
            for p in model['fit_params']:
                if model['fit_params'][p]['bounds'] != None:
                    n_params +=1
            n.append(n_params)
        n = np.array(n)
        
        ### chisq + 2*ln(d) = AIC    
        AIC = np.array([2*np.log(i) + 2*n for i in self.dataset[self.all_chisq_var].values])

        AIC_labels = np.argmin(AIC,axis=1)
        AIC_label_names = np.array([self.dataset[self.model_names_var][i].values for i in AIC_labels])
        

        self.output['AIC'] = xr.DataArray(
            data=AIC,
            dims=[self.sample_dim, self.model_names_var]
        )

        self.output[self._prefix_output("labels")] = xr.DataArray(
            data=AIC_labels,
            dims=[self.sample_dim]
        )
        
        self.output[self._prefix_output("label_names")] = xr.DataArray(
            data=AIC_label_names,
            dims=[self.sample_dim]
        )

        return self


class ModelSelectBIC(PipelineOp):
    def __init__(
        self,
        all_chisq_var,
        model_names_var,
        sample_dim,
        output_prefix='BIC',
        name="ModelSelectionBIC",
        server_id="localhost:5058",
        **kwargs
    ):
        
        output_variables = ["labels", "label_names", "BIC"]
        super().__init__(
            name=name,
            input_variable=[all_chisq_var, model_names_var],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
       
        self.server_id = server_id
        self.sample_dim = sample_dim
        self.model_names_var = model_names_var
        self.all_chisq_var = all_chisq_var
    
    def construct_clients(self):
        """
        creates a client to talk to the AutoSAS server
        """

        self.AutoSAS_client = Client(
            self.server_id.split(":")[0], port=self.server_id.split(":")[1]
        )
        self.AutoSAS_client.login("AutoSAS_client")
        self.AutoSAS_client.debug(False)

    def calculate(self, dataset):        
        """Method for selecting the model based on parsimony given a user defined ChiSq threshold """
        
        self.construct_clients()
        self.dataset = dataset.copy(deep=True)

        bestChiSq_labels = self.dataset[self.all_chisq_var].argmin(self.model_names_var).values
        bestChiSq_label_names = np.array([self.dataset[self.model_names_var][i].values for i in bestChiSq_labels])
        
        aSAS_config = self.AutoSAS_client.get_config('all',interactive=True)['return_val']
        n = []
        for model in aSAS_config['model_inputs']:
        #for model in dataset[self.model_names_var].values:
            n_params = 0
            for p in model['fit_params']:
                if model['fit_params'][p]['bounds'] != None:
                    n_params +=1
            n.append(n_params)
        n = np.array(n)
        
        ###  n*ln(len(q))- 2*ln(chisq) = BIC    
        BIC = np.array([n*np.log(len(self.dataset.q.values)) - 2*np.log(i) for i in self.dataset[self.all_chisq_var].values])

        BIC_labels = np.argmin(AIC,axis=1)
        BIC_label_names = np.array([self.dataset[self.model_names_var][i].values for i in AIC_labels])
        

        self.output['BIC'] = xr.DataArray(
            data=AIC,
            dims=[self.sample_dim, self.model_names_var]
        )

        self.output[self._prefix_output("labels")] = xr.DataArray(
            data=BIC_labels,
            dims=[self.sample_dim]
        )
        
        self.output[self._prefix_output("label_names")] = xr.DataArray(
            data=BIC_label_names,
            dims=[self.sample_dim]
        )

        return self


class ModelSelectBayesianModelComparison(PipelineOp):
    """Uses a Bayesian model comparison approach to calculating probailities given a set of models and outputs"""
    def __init__(
        self,
    ):
        return
    
    def calculate(self):
        return 

