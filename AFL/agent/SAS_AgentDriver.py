from AFL.automation.APIServer.Client import Client
from AFL.automation.prepare.OT2Client import OT2Client
from AFL.automation.shared.utilities import listify
from AFL.automation.APIServer.Driver import Driver
from AFL.automation.shared.Serialize import serialize,deserialize

from math import ceil,sqrt
import json
import time
import requests
import shutil
import datetime
import traceback
import pickle,base64

import pandas as pd
import numpy as np
import xarray as xr
import pathlib

import AFL.agent.PhaseMap
from AFL.agent import AcquisitionFunction 
from AFL.agent import GaussianProcess 
from AFL.agent import Similarity 
from AFL.agent import PhaseLabeler
from AFL.agent.WatchDog import WatchDog


import tensorflow as tf
import gpflow

import uuid

class SAS_AgentDriver(Driver):
    defaults={}
    defaults['compute_device'] = '/device:CPU:0'
    defaults['data_path'] = '~/'
    defaults['data_manifest_file'] = 'manifest.csv'
    defaults['save_path'] = '/home/AFL/'
    defaults['data_tag'] = 'default'
    # defaults['grid_pts_per_row'] = 100
    def __init__(self,overrides=None):
        Driver.__init__(self,name='SAS_AgentDriver',defaults=self.gather_defaults(),overrides=overrides)

        self.watchdog = None 
        self.data_manifest = None
        self._app = None
        self.name = 'SAS_AgentDriver'

        self.status_str = 'Fresh Server!'

        self.phasemap = None
        self.n_cluster = None
        self.similarity = None
        self.next_sample = None
        self.mask = None
        self.iteration = 0
        self.acq_count = 0
        
    @property
    def app(self):
        return self._app
    
    @app.setter
    def app(self,value):
        self._app = value
        # if value is not None:
        #     self.reset_watchdog()
        
    def status(self):
        status = []
        status.append(self.status_str)
        status.append(f'Using {self.config["compute_device"]}')
        status.append(f'Data Manifest:{self.config["data_manifest_file"]}')
        status.append(f'Iteration {self.iteration}')
        status.append(f'Acquisition Count {self.acq_count}')
        return status
    
    def reset_watchdog(self):
        if not (self.watchdog is None):
            self.watchdog.stop()
            
        if self.app is not None:
            logger = self.app.logger
        else:
            logger = None
        
        path = pathlib.Path(self.config['data_manifest_file'])
        self.watchdog = WatchDog(
            path=path.parent,
            fname=path.name,
            callback=self.update_phasemap,
            cooldown=5,
        )
        self.watchdog.start()
        
    def update_status(self,value):
        self.status_str = value
        self.app.logger.info(value)
    
    def get_object(self,name):
        return serialize(getattr(self,name))
    
    def set_mask(self,mask,serialized=False):
        if serialized:
            mask = deserialize(mask)
        self.mask = mask
    
    def set_similarity(self,name,similarity_params):
        if isinstance(name,str):
            if name=='pairwise':
                self.similarity = Similarity.Pairwise(params=similarity_params)
            else:
                raise ValueError(f'Similarity type not recognized:{name}')
        else:
            similarity = deserialize(name)
            self.similarity = similarity

    def set_labeler(self,name):
        if isinstance(name,str):
            if name=='gaussian_mixture_model':
                self.labeler = PhaseLabeler.GaussianMixtureModel()
            else:
                raise ValueError(f'Similarity type not recognized:{name}')
        else:
            labeler = deserialize(name)
            self.labeler = labeler
            
    def set_acquisition(self,spec):
        if isinstance(spec,dict):
            if spec['name']=='variance':
                self.acquisition = AcquisitionFunction.Variance()
            elif spec['name']=='random':
                self.acquisition = AcquisitionFunction.Random()
            elif spec['name']=='combined':
                function1 = spec['function1_name']
                function2 = spec['function2_name']
                function2_frequency= spec['function2_frequency']
                function1 = AcquisitionFunction.Variance()
                function2 = AcquisitionFunction.Random()
                self.acquisition = AcquisitionFunction.IterationCombined(
                    function1=function1,
                    function2=function2,
                    function2_frequency=function2_frequency,
                )
            else:
                raise ValueError(f'Acquisition type not recognized:{name}')
        else:
            acq = deserialize(spec)
            self.acquisition = spec
        
    def append_data(self,data_dict):
        data_dict = {k:deserialize(v) for k,v in data_dict.items()}
        self.phasemap = self.phasemap.append(data_dict)

    def read_data(self,predict=True):
        self.update_status(f'Reading the latest data in {self.config["data_manifest_file"]}')
        path = pathlib.Path(self.config['data_path'])
        
        self.data_manifest = pd.read_csv(path/self.config['data_manifest_file'])


        self.phasemap = xr.Dataset()
        measurements = []
        for i,row in self.data_manifest.iterrows():
            #measurement = pd.read_csv(path/row['fname'],comment='#'.set_index('q').squeeze()
            measurement = pd.read_csv(path/row['fname'],sep=',',comment='#',header=None,names=['q','I']).set_index('q').squeeze().to_xarray()
            measurement.name = row['fname']
            measurements.append(measurement)
        self.phasemap['raw_data'] = xr.concat(measurements,dim='sample')

        compositions = self.data_manifest.drop(['label','fname'],errors='ignore',axis=1)
        self.components = list(compositions.columns.values)
        for component in self.components:
            self.phasemap[component] = ('sample',compositions[component].values)

        
        if 'labels' in self.data_manifest:
            self.phasemap['labels'] = ('sample',self.data_manifest['labels'])
        else:
            self.phasemap = self.phasemap.afl.labels.make_default()

        self.phasemap.attrs['components'] = self.components
        self.phasemap.attrs['components_grid'] = [i+'_grid' for i in self.components]
        self.phasemap['mask'] = self.mask #should add grid dimensions automatically
        #must reset for serlialization to netcdf to work
        self.phasemap = self.phasemap.reset_index('grid').reset_coords(self.phasemap.attrs['components_grid'])

        if predict:
            self.predict()

    def process_data(self,process=True,qlo=0.0001,qhi=0.3,pedestal=1e-12,serialize=False):
        # should this put the q on logscale? Should we resample data to the sample q-values? geomspaced?
        measurements = self.phasemap['raw_data'].copy()
        
        #q-range masking
        q = measurements['q']
        mask = (q>qlo)&(q<qhi)
        measurements = measurements.where(mask,drop=True)
        
        #pedestal + log10 normalization
        measurements += pedestal 
        measurements = np.log10(measurements)
        measurements['q'] = np.log10(measurements['q'])
        measurements = measurements.rename(q='logq')
        
        #fixing Nan to pedestal values
        measurements = measurements.where(~np.isnan(measurements)).fillna(pedestal)
        
        #invariant scaling 
        #norm = measurements.integrate('q')
        #measurements = measurements/norm

        self.phasemap['processed_data'] = measurements
        
        return measurements
    

    def label(self):
        self.update_status('Labelling data on iteration {self.iteration}')
        self.similarity.calculate(self.phasemap['processed_data'])

        ###XXX need to add cutoout for labelers that don't need silhouette or to use other methods
        self.n_cluster,labels,silh = PhaseLabeler.silhouette(self.similarity.W,self.labeler)
        self.update_status(f'Silhouette analysis found {self.n_cluster} clusters')

        self.phasemap.attrs['n_cluster'] = self.n_cluster
        self.phasemap['labels'] = ('sample',labels)
        self.phasemap = self.phasemap.afl.labels.make_ordinal()
        
    def extrapolate(self):
        # Predict phase behavior at each point in the phase diagram
        self.update_status(f'Starting gaussian process calculation on {self.config["compute_device"]}')
        with tf.device(self.config['compute_device']):
            self.GP = GaussianProcess.GP(
                self.phasemap,
                self.components,
                num_classes=self.n_cluster
            )
            kernel = gpflow.kernels.Matern32(variance=0.5,lengthscales=1.0) 
            self.GP.reset_GP(kernel = kernel)          
            self.GP.optimize(2000,progress_bar=True)

        self.acq_count   = 0
        self.iteration  += 1 #iteration represents number of full calculations

        self.update_status(f'Finished AL iteration {self.iteration}')
        
    @Driver.unqueued()
    def get_next_sample(self):
        self.update_status(f'Calculating acquisition function...')
        check = self.data_manifest[self.components].values
        self.acquisition.reset_phasemap(self.phasemap,self.components)
        self.acquisition.reset_mask(self.mask)
        self.phasemap = self.acquisition.calculate_metric(self.GP)

        self.update_status(f'Finding next sample composition based on acquisition function')
        check = self.data_manifest[self.components].values
        self.next_sample = self.acquisition.get_next_sample(composition_check=check)
        self.update_status(f'Next sample is found to be {self.next_sample.squeeze().to_dict()} by acquisition function {self.acquisition.name}')
        self.acq_count+=1#acq represents number of times 'get_next_sample' is called
        
    @Driver.unqueued()
    def save_results(self):
        #write netcdf
        uuid_str = str(uuid.uuid4())
        save_path = pathlib.Path(self.config['save_path'])
        date =  datetime.datetime.now().strftime('%y%m%d')
        time =  datetime.datetime.now().strftime('%H:%M:%S')
        self.phasemap['gp_y_var'] = (('grid','phase_num'),self.acquisition.y_var)
        self.phasemap['gp_y_mean'] = (('grid','phase_num'),self.acquisition.y_mean)
        self.phasemap['next_sample'] = self.next_sample
        self.phasemap.attrs['uuid'] = uuid_str
        self.phasemap.attrs['date'] = date
        self.phasemap.attrs['time'] = time
        self.phasemap.attrs['data_tag'] = self.config["data_tag"]
        self.phasemap.attrs['acq_count'] = self.acq_count
        self.phasemap.attrs['iteration'] = self.iteration
        self.phasemap.to_netcdf(save_path/f'phasemap_{self.config["data_tag"]}_{uuid_str}.nc')
        
        #write manifest csv
        AL_manifest_path = save_path/'manifest.csv'
        if AL_manifest_path.exists():
            self.AL_manifest = pd.read_csv(AL_manifest_path)
        else:
            self.AL_manifest = pd.DataFrame(columns=['uuid','date','time','data_tag','iteration','acq_count'])
        
        row = {}
        row['uuid'] = uuid_str
        row['date'] =  date
        row['time'] =  time
        row['data_tag'] = self.config['data_tag']
        row['iteration'] = self.iteration
        row['acq_count'] = self.acq_count
        self.AL_manifest = pd.concat([self.AL_manifest.T,pd.Series(row)],axis=1,ignore_index=True).T
        self.AL_manifest.to_csv(AL_manifest_path,index=False)
            
    def predict(self):
        self.process_data()
        self.label()
        self.extrapolate()
        self.get_next_sample()
        self.save_results()
    


   
