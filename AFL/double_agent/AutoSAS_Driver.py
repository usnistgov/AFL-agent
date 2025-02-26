import copy
import pathlib
import uuid
from collections import defaultdict
from typing import Dict, Any, Optional
from typing_extensions import Self

from tqdm.auto import tqdm  # type: ignore
import h5py  # type: ignore
import numpy as np
import pandas as pd
import xarray as xr

import bumps  # type: ignore
import bumps.fitproblem  # type: ignore
import bumps.fitters  # type: ignore
import bumps.names  # type: ignore
import sasmodels  # type: ignore
import sasmodels.bumps_model  # type: ignore
import sasmodels.core  # type: ignore
import sasmodels.data  # type: ignore

from AFL.automation.APIServer.Driver import Driver  # type: ignore


class SASModelWrapper:
    """A wrapper class for sasmodels and bump fitting

    Each sasmodel should contain a kernel with the appropriate model type and parameters. Bumps fitting or DREAM
    will require extra methods.


    fit method returns the results object which can be reformatted into the appropriate dictionary structure. It can
    also be called in the current instantiation of the sas_wrapper
    """

    def __init__(
        self,
        name: str,
        data: sasmodels.data.Data1D,
        sasmodel: str,
        parameters: Dict[str, Any],
    ) -> None:
        self.name = name
        self.data = data
        self.sasmodel = sasmodel
        self.kernel = sasmodels.core.load_model(sasmodel)
        self.init_params = copy.deepcopy(parameters)

        # instantiate a bumps model and set the parameter values and bounds
        self.model = sasmodels.bumps_model.Model(self.kernel)
        for key in parameters:
            self.model.parameters()[key].value = parameters[key]["value"]
            if parameters[key]["bounds"] is not None:
                self.model.parameters()[key].fixed = False
                # noinspection PyUnresolvedReferences
                self.model.parameters()[key].bounds = bumps.bounds.Bounded(
                    *parameters[key]["bounds"]
                )

        self.experiment = sasmodels.bumps_model.Experiment(data=data, model=self.model)
        self.problem = bumps.fitproblem.FitProblem(self.experiment)

        self.model_I = None
        self.model_q = None
        self.fit_params = None

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def construct_I(self, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return self(params)

    def residuals(self):
        return self.problem.residuals()

    def __call__(self, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        builds the model from the established or updated parameters
        """
        if params is not None:
            for k, v in params.items():
                self.model.parameters()[k].value = v
            self.experiment.update()
        return self.experiment.theory()

    def fit(self, fit_method=None):
        if fit_method is None:
            fit_method = {
                "method": "lm",
                "steps": 1000,
                "ftol": 1.5e-6,
                "xtol": 1.5e-6,
                "verbose": True,
            }
        # self.results = bumps.fitters.fit(self.problem,**fit_method)
        try:
            self.results = bumps.fitters.fit(self.problem, **fit_method)
        except:
            self.results = bumps.fitters.fit(
                self.problem,
                method="lm",
                steps=1000,
                ftol=1.5e-6,
                xtol=1.5e-6,
                verbose=True,
            )

        self.fit_params = {
            key: val for key, val in zip(self.problem.labels(), self.problem.getp())
        }
        self.model_I = self.construct_I(params=self.fit_params)
        self.model_q = self.data.x[self.data.mask == 0]
        self.model_cov = self.problem.cov()
        return self.results

    def get_fit_params(self):
        params = copy.deepcopy(self.init_params)
        for idx, r in enumerate(self.problem.labels()):
            params[r] = {}
            params[r]["value"] = self.results.x[idx]
            params[r]["error"] = self.results.dx[idx]

        for key in list(params):
            if "bounds" in list(params[key]):
                params[key]["error"] = params[key]["bounds"]
                del params[key]["bounds"]
        return params


class AutoSAS_Driver(Driver):
    defaults = {}
    defaults["savepath"] = "/home/afl642/2402_DT_ISIS_path"
    defaults["q_min"] = 1e-2
    defaults["q_max"] = 1e-1
    defaults["resolution"] = None
    defaults["model_inputs"] = [
        {
            "name": "power_law_1",
            "sasmodel": "power_law",
            "q_min":0.01,
            "q_max":0.4,
            "fit_params": {
                "power": {"value": 4, "bounds": (3, 4.5)},
                "background": {"value": 1e-4, "bounds": (1e-10, 1e2)},
                "scale": {"value": 1e0, "bounds": (1e-6, 1e4)},
            },
        }
    ]
    defaults["fit_method"] = {
        "method": "lm",
        "steps": 1000,
        "ftol": 1.5e-6,
        "xtol": 1.5e-6,
        "verbose": True,
        "test_var": "new",
    }

    def __init__(self):
        Driver.__init__(self, name="SAS_model_fitter", defaults=self.gather_defaults())
        self.models = []
        self.models_post = []
        self.results = []
        self.report = {}
        self.status_str = "Fresh Server!"
        self.dropbox = dict()

        self.sasdata = []
        self.model_fit = False
        print("self.data exists == :",self.data)

    def status(self):
        status = []
        status.append(self.status_str)
        return status

    def update_status(self, status):
        self.status_str = status
        if self.app is not None:
            self.app.logger(status)

    def construct_models(self, data: sasmodels.data.Data1D) -> None:
        """
        This works off of the persistent config and will generate a list of sas_wrapper models containing the
        kernels, the experiments, the problems and resolution etc.
        """
        self.models = []
        for inputs in self.config["model_inputs"]:
            q_min = inputs.get("q_min", self.config["q_min"])
            q_max = inputs.get("q_max", self.config["q_max"])
            data.mask = (data.x < q_min) | (data.x > q_max)
            print(data.y.shape,data.y)
            model = SASModelWrapper(
                name=inputs["name"],
                data=copy.deepcopy(data),
                sasmodel=inputs["sasmodel"],
                parameters=inputs["fit_params"],
            )
            self.models.append(model)

        self.models_fit = False

    def store_results(self, model_list=None, filetype=None):
        """stores the results of the fitting into the appropriate structure and filetype and push it to the tiled
        server"""
        results_list = []
        if model_list:
            model_list = model_list
        else:
            model_list = self.models

        if self.models_fit:
            for model in model_list:
                name = model.name
                sasmodel = model.sasmodel
                chi_sqr = model.problem.chisq()
                cov = model.model_cov

                # new parameters needs to take on the same form as the input model but now it no longer has bounds
                # but error
                params = model.init_params
                for idx, r in enumerate(model.problem.labels()):
                    params[r] = {}
                    params[r]["value"] = model.results.x[idx]
                    params[r]["error"] = model.results.dx[idx]

                for key in list(params):
                    if "bounds" in list(params[key]):
                        params[key]["error"] = params[key]["bounds"]
                        del params[key]["bounds"]

                results_list.append(
                    {
                        "name": name,
                        "sasmodel": sasmodel,
                        "chisq": chi_sqr,
                        "cov"  : cov.tolist(),
                        "output_fit_params": params,
                    }
                )
        else:
            pass
        self.models_post = []
        return results_list

    def set_sasdata(
        self,
        db_uuid: str,
        sample_dim: str = "sample",
        q_variable: str = "q",
        sas_variable: str = "I",
        sas_err_variable: str = "dI",
        sas_resolution_variable: Optional[str] = None,
    ) -> None:
        """
        Set the sasdata to be fit

        Parameters
        ----------
        db_uuid: str
            Dropbox UUID to retrieve `xarray.Dataset` from. The Dataset should be deposited using `Client.deposit_obj`
            in interactive mode in order to obtain the uuid of the deposited item. See example below

        sample_dim: str
            The `xarray` dimension containing each sample

        q_variable: str
            The name of the `xarray.Dataset` variable corresponding to the q (wavevector, wavenumber) values

        sas_variable: str
            The name of the `xarray.Dataset` variable corresponding to the scattering intensity to be fit

        sas_err_variable: str
            The name of the `xarray.Dataset` variable corresponding to the uncertainty in the scattering intensity

        sas_resolution_variable: Optional[str]
            The name of the `xarray.Dataset` variable corresponding to the resolution function


        Example
        -------
        ```python
        from AFL.automation.APIServer.Client import Client
        import xarray as xr

        client = Client('localhost',port=5058)
        client.login('User')
        db_uuid = client.deposit_obj(obj=xr.Dataset())
        client.enqueue(task_name='initialize_input',db_uuid=db_uuid)
        ```
        """
        self.sasdata_dataset = self.retrieve_obj(db_uuid)

        self.sasdata = []
        for i, sds in self.sasdata_dataset.groupby(sample_dim, squeeze=False):
            x = sds[q_variable].squeeze().values
            y = sds[sas_variable].squeeze().values
            dy = sds[sas_err_variable].squeeze().values
            if sas_resolution_variable is not None:
                dx = sds[sas_resolution_variable].values
            else:
                dx = None

            self.sasdata.append(sasmodels.data.Data1D(x=x, y=y, dy=dy, dx=dx))

    def fit_models(self, parallel=False, model_list=None, fit_method=None):
        """
        Executes a fit models call

        Parameters
        ----------
        parallel: bool
            NOT IMPLEMENTED!!! executes a multi-process fitting call to speed up the data
        
        fit_method: Optional[dict]
            a dictionary defining the fitting routine to pass into the bumps fitting routine
            
        model_list: Optional[list]
            the list of formatted dictionaries for the SAS models construction        
        """

        if not self.sasdata:  # empty
            raise ValueError(
                "No sasdata to fit! Use .set_sasdata(...) to initialize data to be fit!"
            )

        self.results = []
        if fit_method is None:
            fit_method = self.config["fit_method"]
        else:
            self.config["fit_method"] = fit_method

        if model_list is None:
            model_list = self.models

        fitted_models = []
        # for data in tqdm(self.sasdata, total=len(self.sasdata)):
        for idx,data in enumerate(self.sasdata):
            self.models_post = []
            self.construct_models(data)

            fitted_models.append([])
            for model in self.models:
                print('SASDATA IDX:',idx,'MODEL:',model)
                model.fit(fit_method=fit_method)
                self.models_post.append(model)
                self.models_fit = True

                fitted_models[-1].append(model.copy())

            self.results.append(self.store_results(self.models_post))

        self.build_report()

        # construct array of theory fits
        tiled_arrays = defaultdict(list)
        for fitted_models in fitted_models:
            for model in fitted_models:
                tiled_arrays[f"chisq_{model.name}"].append(model.problem.chisq())
                tiled_arrays[f"fit_I_{model.name}"].append(model.model_I)
                tiled_arrays[f"residuals_{model.name}"].append(model.residuals())
                if len(tiled_arrays[f'fit_q_{model.name}'])==0:
                    tiled_arrays[f'fit_q_{model.name}'] = model.model_q

                tiled_arrays[f"params_{model.name}"].append(model.get_fit_params())


        #duplicating for dropbox
        # dropbox_dict= dict(tiled_arrays)

        as_uuid = 'AS-' + str(uuid.uuid4())
        
        # construct arrays and save to tiled
        # self.data.add_array('probabilities', self.report['probabilities'])
        self.data.add_array("best_chisq", self.report["best_fits"]["lowest_chisq"])
        self.data.add_array("model_names", self.report["best_fits"]["model_name"])
        self.data.add_array("all_chisq", self.report["all_chisq"])
        self.data.add_array("probabilities", self.report["probabilities"])

        # dropbox_dict["best_chisq"] = self.report["best_fits"]["lowest_chisq"]
        # dropbox_dict["model_names"] = self.report["best_fits"]["model_name"]
        # dropbox_dict["all_chisq"] = self.report["all_chisq"]
        
        for array_name, array in tiled_arrays.items():
            if "params_" in array_name:
                df_value = pd.DataFrame([
                    {k:v['value'] for k,v in item.items()}
                    for item in array
                ])
                df_error = pd.DataFrame([
                    {k:v['error'] for k,v in item.items()}
                    for item in array
                ])

                df_value = df_value.fillna(-1)
                df_error = df_error.fillna(-1)

                self.data[array_name+"_columns"] = df_value.columns.values
                self.data.add_array(array_name,df_value.values)
                self.data.add_array(array_name+'_error',df_error.values)
                
                # dropbox_dict[array_name] = df_value.values
                # dropbox_dict[array_name+'_error'] = df_error.values

            else:
                self.data.add_array(array_name, np.array(array))

                # dropbox_dict[array_name] = np.array(array)
        # print('AFTER TILED2')




        ####################################################################################################
        ### fixing the write so that it stores not just to tiled but to dropbox
        ####################################################################################################
        self.sample_dim = "sas_fit_sample"
        self.model_dim = "models"
        
         
        dso = xr.Dataset()
        print(self.sample_dim, self.model_dim)
        print()
        dso[("all_chisq")] = xr.DataArray(
            data = self.report["all_chisq"], 
            dims = [self.sample_dim, self.model_dim],
            coords = {self.sample_dim: np.arange(len(self.sasdata)), self.model_dim: [m["name"] for m in list(self.report["model_fits"][0])]}
        )


        ### this needs to get moved to the pipeline object
        dso[("all_chisq")].attrs[
            "fit_calc_id"
        ] = as_uuid

        dso[("probabilities")] = xr.DataArray(
            data = self.report['probabilities'],
            dims = [self.sample_dim, self.model_dim],
            coords = {self.sample_dim: np.arange(len(self.sasdata)), self.model_dim: [m["name"] for m in list(self.report["model_fits"][0])]}
        )


        #build from the report json
        target = {}
        err = {}
        for idx,fit in enumerate(self.report["model_fits"]):
            for model in fit:
                target[model['name']] = {}
                err[model['name']] = {}
                
                for param in model["output_fit_params"]:
                    param_name = '_'.join([model['name'],param])
                    target[model['name']][param_name] = []
                    err[model['name']][param_name] = []
                    
        for idx,fit in enumerate(self.report["model_fits"]):
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
            dso[f"{key}_fit_val"] = xr.DataArray(
                data=p_data,
                coords={model_dim:list(target[key])},
                dims=[model_dim,self.sample_dim]
            )
            dso[f"{key}_fit_val"].attrs[
                "fit_calc_id"
            ] = as_uuid

            e_data = np.array([err[key][param] for param in list(err[key])])
            
            dso[f"{key}_fit_err"] = xr.DataArray(
                data=e_data,
                coords=[list(target[key]), np.arange(e_data.shape[1])],
                dims=[model_dim, self.sample_dim]
            )
            dso[f"{key}_fit_err"].attrs[
                "fit_calc_id"
            ] = as_uuid

        

          
        
        model_ids = list(target)
        
        for idx, mid in enumerate(model_ids):
            q_key = [i for i in list(tiled_arrays) if ('fit_q' in i and mid in i)][0]
            print(mid)
            
            for key in list(tiled_arrays):
        
                if (mid in key) and ('fit_I' in key):
                    dso[key] = xr.DataArray(
                        tiled_arrays[key],
                        dims=[self.sample_dim,q_key],
                        coords=[np.arange(e_data.shape[1]), tiled_arrays[q_key]]
                    )
                elif (mid in key) and ('residuals' in key):
                    dso[key] = xr.DataArray(
                        tiled_arrays[key],
                        dims=[self.sample_dim,q_key],
                        coords=[np.arange(e_data.shape[1]), tiled_arrays[q_key]]
                    )
        
        ##################################################

        print('stored to dropbox')
        self.deposit_obj(obj=dso,uid=as_uuid)

                
        ####################
        # a main process has to exist for this to run. not sure how it should interface on a server...
        ####################
        # if parallel:
        #     processes = []
        #     for model in model_list:
        #         p = Process(target=model.fit(),)
        #         p.start()
        #         processes.append(p)
        #     process_states = [False for i in process]
        #     while all(process_states) == False:
        #         print('not all models have converged')
        #         time.sleep(1)
        #         for idx, p in enumerate(processes):
        # else:
        #     for model in model_list:
        #         model.fit(data=data, fit_method=fit_method)
        return as_uuid

    def build_report(self):
        """
        Builds a human readable report for the fitting results.
        TODO: Want a readable PDF built up from FPDF...
        """
        print("BUILDING REPORT!")
        self.report["fit_method"] = self.config["fit_method"]
        self.report["model_inputs"] = self.config["model_inputs"]
        # self.report['best_model'] =
        self.report["model_fits"] = self.results
        bf = {}
        best_chis = []
        best_names = []
        indices = []
        best_models = []
        for idx, result in enumerate(self.results):
            chisqs = [model["chisq"] for model in result]
            names = [model["name"] for model in result]

            i = np.nanargmin(chisqs)
            best_chis.append(chisqs[i])
            best_names.append(names[i])
            indices.append(i)
        #print(best_chis, best_names, indices)
        bf["model_name"] = best_names
        bf["lowest_chisq"] = best_chis
        bf["model_idx"] = [int(i) for i in indices]
        bf["model_params"] = [
            self.report["model_fits"][idx][m] for idx, m in enumerate(indices)
        ]
        self.report['probabilities'] = self.calc_probabilities()
        self.report['all_chisq'] = [[model['chisq'] for model in result] for result in self.results]
        self.report["best_fits"] = bf

        print("REPORT BUILT!")

        return self.report

    def calc_probabilities(self):
        """
        Calculates the probability of the set of models given the following approximations
        
        Uses the loss function (chisq) as an estimate of the model log likelihood p(t|M(i),theta) and the Laplace approximation to construct a marginal log likelihood.
        This relies on gaussain curvature (Hessian) which in the case of multivariate normal distributions is the inverse of the covariance matrix. 
          
        """
        
        all_chisqs = []
        all_fit_params = []
        all_fit_cov = []

        marginal_likelihoods = []
        print(len(self.models), len(self.results[0]))
        for result in self.results:
            
            model_likelihood = [-1*model['chisq'] for model in result]
            #d = len(model.fit_params)
            d = [len(model['cov']) for model in result]
            cov = [np.array(model['cov']) for model in result]
            log_marginal_likelihood = [model_likelihood[i] + 0.5*np.log(np.linalg.det(cov[i])) + 0.5*d[i]*np.log(2*np.pi) for i in range(len(d))]
            
            #print(model_likelihood,d,log_marginal_likelihood)
            marginal_likelihoods.append(np.exp(log_marginal_likelihood))

        all_probs = []
        for row in marginal_likelihoods:
            probs = [i/sum(row) for i in row]
            #print(probs)
            #print(sum(probs))
            all_probs.append(probs)

        #model_probabilities = [j/sum(i) for i in log_marginal_likelihoods for j in i]
       # self.model_probabilities = np.array(model_probabilities).reshape(len(self.results),len(d))
        self.model_probabilities = np.array(all_probs)
        return self.model_probabilities

    
    def model_selection(self, chisqr_tol=1e0):
        """
        This returns the model selected based either on BIC or some other metric for which one is correct

        should return the model name, the parameters that were optimized as well as the flags? and chi squared?
        One could in theory do the crystalshift tree search method to help with the complexity of multi-model fitting
        """

        if self.models_fit:
            print("returning the ideal model")
        else:
            print("the models have not been fit. run the fit_model method")

        all_chisqr = [model.problem.chisq() for model in self.models_post]
        names = [model.name for model in self.models_post]
        all_results = [model.results for model in self.models_post]

        # check if the chisqr is bad for all...
        if np.all(all_chisqr > chisqr_tol):
            print(
                "all models do not meet the fitting criteria! try fitting with different starts, call DREAM, "
                "or add a model"
            )
        else:
            model_idx = np.nanargmin(all_chisqr)
            selected_model = self.models[model_idx]
            print(f"best model is {selected_model.name}")

        # the BIC criteria can be supplemented here. as of now it is a argmin of the chisqr
        return selected_model.name

    def _writedata(self, data):
        filename = pathlib.Path(self.config["filename"])
        filepath = pathlib.Path(self.config["filepath"])
        print(f"writing data to {filepath/filename}")
        with h5py.File(filepath / filename, "w") as f:
            f.create_dataset(str(uuid.uuid1()), data=data)

_DEFAULT_PORT = 5058
if __name__ == '__main__':
    from AFL.automation.shared.launcher import *
