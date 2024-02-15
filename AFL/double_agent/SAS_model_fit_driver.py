# import sasmodels
import sasmodels
import sasmodels.core
import sasmodels.data
import sasmodels.bumps_model

# import bumps
import bumps
import bumps.fitters
import bumps.names
import bumps.fitproblem

# import sasview
import copy as cp

import numpy as np
import os
import pandas as pd
import periodictable
import periodictable.nsf
# import plotly.express as px # plotting
import re
import sys
# import tol_colors as tc # colorblind safe color palettes
import xarray as xr

import time
import datetime
from AFL.automation.APIServer.Driver import Driver
import h5py
import pathlib
import uuid
import tensorflow
from multiprocessing import Process, Pool
import tensorflow as tf



class sas_wrapper():
    """A wrapper class for sasmodels and bump fitting
    
    Each sasmodel should contain a kernel with the appropriate model type and parameters. Bumps fitting or DREAM
    will require extra methods.
    
    
    fit method returns the results object which can be reformatted into the appropriate dictionary structure. It can also be called in the current instantiation of the sas_wrapper
    """
    def __init__(self, name, sasmodel_type, parameters, empty_data, resolution=None):
        
        #build the sasmodel from the start
        self.sasmodel_type = sasmodel_type
        self.kernel = sasmodels.core.load_model(sasmodel_type)
        self.call_kernel = sasmodels.direct_model.DirectModel(empty_data,self.kernel)
        self.call_kernel.resolution = resolution
        self.name = name
        self.init_params = cp.deepcopy(parameters)
        
        #pull the parameters from the initialization dictionary and create a model
        model_params = {}
        for key in parameters:
            model_params[key] = parameters[key]['value']
        self.model = sasmodels.bumps_model.Model(self.kernel, **model_params)
        
        #augment the bounds. clunky but it works
        for key in parameters:
            
            bounds = parameters[key]['bounds']
            if bounds != None:
                self.model.parameters()[key].fixed = False
                self.model.parameters()[key].bounds = bumps.bounds.Bounded(bounds[0],bounds[1])
        # print(model_params)
        self.model_I = self.construct_I(params=model_params)
        self.model_q = empty_data.x
                
    def fit(self, data, fit_method=None):
        self.experiment = sasmodels.bumps_model.Experiment(data=data, model=self.model)
        self.problem = bumps.fitproblem.FitProblem(self.experiment)
        if fit_method==None:
            print('no specified fit method. Using a LM default')
            fit_method = {'method':'lm',
                          'steps':1000,
                          'ftol': 1.5e-6,
                          'xtol': 1.5e-6,
                          'verbose':True
                         }
        # self.results = bumps.fitters.fit(self.problem,**fit_method)
        try:
            print('trying to fit with **fit_method')
            self.results = bumps.fitters.fit(self.problem,**fit_method)
        except:
            print("couldn't read the fit method. Using defaults")
            self.results = bumps.fitters.fit(self.problem, method='lm', steps=1000, ftol=1.5e-6, xtol=1.5e-6, verbose=True)
            
        # print(self.problem)
        fit_params = {key:val for key,val in zip(self.problem.labels(),self.problem.getp())}
        self.model_I = self.construct_I(params=fit_params)
        return self.results
    
    def construct_I(self, params=None):
        """
        builds the model from the established parameters
        """
        model_I = self.call_kernel(**params)
        return model_I
        
class SAS_model_fit(Driver):
    defaults = {}
    defaults['savepath'] = '/home/afl642/2402_DT_ISIS_path'
    defaults['q_range']  = (1e-2,0.5e-1)
    defaults['n_q_fit'] = 500
    defaults['resolution'] = None
    defaults['model_inputs'] = [{
        'name':'power_law_1',
        'sasmodel':'power_law',
        'fit_params':{
            'power':{
                'value':4,
                'bounds':(3,4.5)
            },
            'background':{
                'value':1e-4,
                'bounds':(1e-10,1e2)
            },
            'scale':{
                'value':1e0,
                'bounds':(1e-6,1e4)
            }
        }
    }]
    defaults['fit_method'] = {
        'method':'lm',
        'steps':1000,
        'ftol': 1.5e-6,
        'xtol': 1.5e-6,
        'verbose':True,
        'test_var':'new'
    }
    def __init__(self):
        Driver.__init__(self,name='SAS_model_fitter',defaults=self.gather_defaults())
        #load the experiment info from the persistent config
        self.q_min = self.config["q_range"][0]
        self.q_max = self.config["q_range"][1]
        self.num_q = self.config["n_q_fit"]
        self.resolution = self.config['resolution']
        self.q_fit = np.logspace(np.log10(self.q_min), np.log10(self.q_max), self.num_q)
        self.empty_data = sasmodels.data.empty_data1D(self.q_fit)
        self.models = []
        self.models_post = []
        self.results = []
        self.report = {}
        # print(self.config)
        
        self.resolution = sasmodels.resolution.Pinhole1D(self.empty_data.x, self.empty_data.x*0.15)
        
        ### persistent config model loading. the dictionary 
        ### spawn a sasmodel for each model in the persistent config
        try:
            self.construct_models()
        except:
            raise ValueError("info is not correct in config for creating models")
        
    def construct_models(self):
        """This works off of the presistent config and will generate a list of sas_wrapper models containing the kernels, the experiments, the problems and resolution etc."""
        self.models=[]
        
        print("")
        print("constructing models from inputs agian")
        for inputs in self.config['model_inputs']:
            print(inputs)
            model = sas_wrapper(
                name = inputs['name'],
                sasmodel_type = inputs['sasmodel'],
                empty_data = self.empty_data,
                parameters = inputs['fit_params'],
                resolution = self.resolution
            )
            self.models.append(model)
        print()
        print(self.models)
        print("")
        print("constructing sas models from persistent config")
        print(f"there are {len(self.models)} potential models")
        for model in self.models:
            print(model.name)
            
        self.models_fit = False
        
        return
    
    def print_model_pointer(self):
        for model in self.models:
            print(hex(id(model)))

                
    def store_results(self,model_list=None, filetype=None):
        """stores the results of the fitting into the appropriate structure and filetype and push it to the tiled server"""
        results_list = []
        if model_list:
            model_list = model_list
        else:
            model_list = self.models
            
        if self.models_fit:
            print('models have been fit, building results structure')
            print(f'there are {len(self.models_post)} models that have been fit')
            for model in model_list:
                print(f'model object location {model}')
                name = model.name
                sasmodel_type = model.sasmodel_type
                # print(model.__dict__)
                chi_sqr = model.problem.chisq()
                
                #new parameters needs to take on the same form as the input model but now it no longer has bounds but error
                params = model.init_params
                print('initial params')
                print(params)
                for idx, r in enumerate(model.problem.labels()):
                    print(r, model.results.x[idx])
                    params[r] = {}
                    params[r]['value'] = model.results.x[idx]
                    params[r]['error'] = model.results.dx[idx]
                    
                for key in list(params):
                    if 'bounds' in list(params[key]):
                        params[key]['error'] = params[key]['bounds']
                        del params[key]['bounds']
                    
                print('replaced params')
                print(params)
                print()
                results_list.append(
                    {'name':name,
                     'sasmodel':sasmodel_type,
                     'chisq':chi_sqr,
                     'output_fit_params':params}
                                     )
        else:
            print('models have not been fit yet. execute the fit_models method')
        #print(results_list)
        self.models_post=[]
        return results_list
    
    
    def fit_models(self,q, I, dI,data_ID=None,parallel=False,model_list=None, fit_method=None):
        """This will fit each available sas_wrapper model in the model list to the data supplied
        q is a list of q vectors
        I is a list of lists of the scattering patterns (n lists of q vectors)
        dI is a list of lists of the uncertainty in the scattering patterns (n lists of q vectors)
        
        data_ID can be a list of string identifiers for each sasview data object
        fit_method is defaulted to in the input but if supplied must be a dictionary of kwargs that pass into the sasmodel.problem method. 
        """
        
        # data = [np.array(i) for i in data]
        #construct the sasdata_1d object for bumps
        sasdata = [sasmodels.data.Data1D(x=np.array(q),
                                         y=np.array(I)[i],
                                         dy=np.array(dI)[i]) for i in range(len(I))]
        # self.data = data
        # self.data_ID = data_ID
        
        self.results=[]
        if fit_method==None:
            print('using default config fitting method')
            fit_method = self.config['fit_method']
        else:
            print(f'using input method {fit_method}')
            self.config['fit_method'] = fit_method
            
        if model_list==None:
            model_list = self.models
            print(f"current self.models{self.models}")
            # print(self.models)
            
        for ddx, d in enumerate(sasdata):
            self.models_post = []
            self.construct_models()
            
            for model in self.models:
                print("starting fitting")
                print(model)
                print()
                model.fit(data=d, fit_method=fit_method)
                self.models_post.append(model)
                self.models_fit = True
                
            self.results.append(self.store_results(self.models_post))
            print('')
            print('')
        print("building report")
        self.build_report()

        ## push to tiled
        self.data.add_array('chisq',self.report['best_fits']['lowest_chisq'])
        self.data.add_array('model_names',self.report['best_fits']['model_name'])
        self.data['report'] = self.report

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
        return

    def build_report(self):
        """
        Builds a human readable report for the fitting results. 
        TODO: Want a readable PDF built up from FPDF...
        """
        self.report['fit_method'] = self.config['fit_method']
        self.report['model_inputs'] = self.config['model_inputs']
        # self.report['best_model'] = 
        print(f"there are {len(self.results)} results")
        print("")
        self.report['model_fits'] = self.results
        bf = {}
        best_chis = []
        best_names = []
        indices = []
        best_models = []
        for idx, result in enumerate(self.results):
            print(idx,result)
            chisqs = [model['chisq'] for model in result]
            names =  [model['name'] for model in result]
            
            i = np.nanargmin(chisqs)
            best_chis.append(chisqs[i])
            best_names.append(names[i])
            indices.append(i)
        print(best_chis,best_names,indices)
        bf['model_name'] = best_names
        bf['lowest_chisq'] = best_chis
        bf['model_idx'] = [int(i) for i in indices]
        bf['model_params'] = [self.report['model_fits'][idx][m] for idx,m in enumerate(indices)]
                
        self.report['best_fits'] = bf
        
        return self.report
        
    def model_selection(self,chisqr_tol=1e0):
        """
        Moved to the labeler class... this server only fits
        
        This returns the model selected based either on BIC or some other metric for which one is correct
        
        should return the model name, the parameters that were optimized as well as the flags? and chi squared?   
        One could in theory do the crystalshift tree search method to help with the complexity of multi-model fitting"""
        
        if self.models_fit:
            print("returning the ideal model")
        else:
            print("the models have not been fit. run the fit_model method")
            
        all_chisqr = [model.problem.chisq() for model in self.models_post]
        names = [model.name for model in self.models_post]
        all_results = [model.results for model in self.models_post]
        
        #check if the chisqr is bad for all...
        if np.all(all_chisqr>chisqr_tol):
            print("all models do not meet the fitting criteria! try fitting with different starts, call DREAM, or add a model")
        else:    
            model_idx = np.nanargmin(all_chisqr)
            selected_model = self.models[model_idx]
            print(f"best model is {selected_model.name}")
            
        #the BIC criteria can be supplemented here. as of now it is a argmin of the chisqr
        return selected_model.name
    
    
    
    def add_model(self,model_dict):
        """ Adds a sas_wrapper model to the list of model objects"""
        print(f"there are {len(self.config['model_inputs'])} potential models")
        print(f"adding a new model")
        for newmodel in model_dict:
            self.config['model_inputs'].append(newmodel)
            self.models.append(sas_wrapper(
                name = newmodel['name'],
                sasmodel_type = newmodel['sasmodel'],
                empty_data = self.empty_data,
                parameters = newmodel['fit_params'],
                resolution = self.resolution
            ))
        self.construct_models
        print(f"there are {len(self.models)} potential models")
        for model in self.models:
            print(model.name)
        self.models_fit = False
            
    def remove_model(self, name):
        """removes a model from the server model list given the colloquial name"""
        for model in self.config['model_inputs']:
            if model.name == name:
                print(f'removing model {name}')
                self.models.remove(model)
        
        for model in self.models:
            print(model.name)
            
    def _writedata(self,data):
        filename = pathlib.Path(self.config['filename'])
        filepath = pathlib.Path(self.config['filepath'])
        print(f'writing data to {filepath/filename}')
        with h5py.File(filepath/filename, 'w') as f:
            f.create_dataset(str(uuid.uuid1()), data=data)