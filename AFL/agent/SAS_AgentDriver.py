from AFL.automation.APIServer.Client import Client
from AFL.automation.prepare.OT2Client import OT2Client
from AFL.automation.shared.utilities import listify
from AFL.automation.APIServer.Driver import Driver
from AFL.automation.shared import serialization
from AFL.automation.shared.utilities import mpl_plot_to_bytes

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

import io
import matplotlib.pyplot as plt
import matplotlib

import AFL.agent.PhaseMap
from AFL.agent import AcquisitionFunction 
from AFL.agent import GaussianProcess 
from AFL.agent import Metric 
from AFL.agent import PhaseLabeler
from AFL.agent.WatchDog import WatchDog


import tensorflow as tf
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.45
## config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

import gpflow

import uuid

class SAS_AgentDriver(Driver):
    defaults={}
    defaults['compute_device'] = '/device:CPU:0'
    defaults['data_path'] = '~/'
    defaults['data_manifest_file'] = 'manifest.csv'
    defaults['save_path'] = '/home/AFL/'
    defaults['data_tag'] = 'default'
    defaults['qlo'] = 0.001
    defaults['qhi'] = 1
    # defaults['grid_pts_per_row'] = 100
    def __init__(self,overrides=None):
        Driver.__init__(self,name='SAS_AgentDriver',defaults=self.gather_defaults(),overrides=overrides)

        self.watchdog = None 
        self.data_manifest = None
        self._app = None
        self.name = 'SAS_AgentDriver'

        self.status_str = 'Fresh Server!'

        self.phasemap = None
        self.n_phases = None
        self.metric = Metric.Dummy()
        self.acquisition = None
        self.labeler = None
        self.next_sample = None
        self._mask = None
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
        if self.metric is not None:
            status.append(f'Metric: {self.metric.name}')
        else:
            status.append(f'Metric: No metric loaded')
        if self.acquisition is not None:
            status.append(f'Acq.: {self.acquisition.name}')
        else:
            status.append(f'Acq.: No acquisition function loaded')
        if self.labeler is not None:
            status.append(f'Labeler: {self.labeler.name}')
        else:
            status.append(f'Labeler: No labeler loaded')
        if self.mask is not None:
            status.append(f'Masking: {self.masked_points}/{self.total_points}')
        else:
            status.append(f'Masking: No mask loaded')
        if self.n_phases is not None:
            status.append(f'Found {self.n_phases} phases')
            
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
    
    # def get_object(self,name):
    #     self.app.logger.info(f'Getting value for {name}')
    #     return serialize(getattr(self,name))
    # 
    # def set_object(self,**kw):
    #     for k,v in kw.items():
    #         self.app.logger.info(f'Setting value for {k}')
    #         setattr(self,k,deserialize(v))
    
    def set_mask(self,mask,serialized=False):
        if serialized:
            mask = serialization.deserialize(mask)
        self.mask = mask
        self.masked_points = int(mask.sum().values)
        self.total_points = mask.size
    
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self,value):
        self._mask=value
        self.masked_points = int(value.sum().values)
        self.total_points = value.size
    
    def set_metric(self,value,**kw):
        if isinstance(value,str):
            if value=='pairwise':
                self.metric = Metric.Similarity(params=kw)
            else:
                raise ValueError(f'Metric type not recognized:{value}')
        else:
            metric = serialization.deserialize(value)
            self.metric = metric

    def set_labeler(self,value):
        if isinstance(value,str):
            if value=='gaussian_mixture_model':
                self.labeler = PhaseLabeler.GaussianMixtureModel()
            else:
                raise ValueError(f'Labeler type not recognized:{value}')
        else:
            labeler = serialization.deserialize(value)
            self.labeler = labeler
            
    def set_acquisition(self,value):
        if isinstance(value,dict):
            if value['name']=='variance':
                self.acquisition = AcquisitionFunction.Variance()
            elif value['name']=='random':
                self.acquisition = AcquisitionFunction.Random()
            elif value['name']=='combined':
                function1 = value['function1_name']
                function2 = value['function2_name']
                function2_frequency= value['function2_frequency']
                function1 = AcquisitionFunction.Variance()
                function2 = AcquisitionFunction.Random()
                self.acquisition = AcquisitionFunction.IterationCombined(
                    function1=function1,
                    function2=function2,
                    function2_frequency=function2_frequency,
                )
            else:
                raise ValueError(f'Acquisition type not recognized:{value}')
        else:
            acq = serialization.deserialize(value)
            self.acquisition = acq
        
    def append_data(self,data_dict):
        data_dict = {k:serialization.deserialize(v) for k,v in data_dict.items()}
        self.phasemap = self.phasemap.append(data_dict)

    def read_data(self):
        self.update_status(f'Reading the latest data in {self.config["data_manifest_file"]}')
        path = pathlib.Path(self.config['data_path'])
        
        self.data_manifest = pd.read_csv(path/self.config['data_manifest_file'])

        self.phasemap = xr.Dataset()
        raw_data_list = []
        data_list = []
        deriv1_list = []
        deriv2_list = []
        fname_list = []
        for i,row in self.data_manifest.iterrows():
            #measurement = pd.read_csv(path/row['fname'],comment='#'.set_index('q').squeeze()
            measurement = pd.read_csv(path/row['fname'],sep=',',comment='#',header=None,names=['q','I'],usecols=[0,1]).set_index('q').squeeze().to_xarray()
            measurement = measurement.dropna('q')

            data   = measurement.afl.scatt.clean(derivative=0,qlo=self.config['qlo'],qhi=self.config['qhi'])
            deriv1 = measurement.afl.scatt.clean(derivative=1,qlo=self.config['qlo'],qhi=self.config['qhi'])
            deriv2 = measurement.afl.scatt.clean(derivative=2,qlo=self.config['qlo'],qhi=self.config['qhi'])

            data   = data - data.min('logq')#put all of the data on the same baseline

            data.name = row['fname']
            deriv1.name = row['fname']
            deriv2.name = row['fname']
            raw_data_list.append(measurement.rename(q='rawq'))
            data_list.append(data)
            deriv1_list.append(deriv1)
            deriv2_list.append(deriv2)
            fname_list.append(row['fname'])
        self.phasemap['fname']   = ('sample',fname_list)
        self.phasemap['raw_data']= xr.concat(raw_data_list,dim='sample')
        self.phasemap['data']    = xr.concat(data_list,dim='sample').bfill('logq').ffill('logq')
        self.phasemap['deriv1']  = xr.concat(deriv1_list,dim='sample').bfill('logq').ffill('logq')
        self.phasemap['deriv2']  = xr.concat(deriv2_list,dim='sample').bfill('logq').ffill('logq')

        # compositions = self.data_manifest.drop(['label','fname'],errors='ignore',axis=1)
        self.components = []
        for column_name in self.data_manifest.columns.values:
            if 'AL_mfrac' in column_name:
                self.components.append(column_name.replace("AL_mfrac_",""))
                self.phasemap[self.components[-1]] = ('sample',self.data_manifest[column_name].values)
            else:
                self.phasemap[column_name] = ('sample',self.data_manifest[column_name].values)
                
        if 'labels' not in self.phasemap:
            self.phasemap = self.phasemap.afl.labels.make_default()

        self.phasemap.attrs['components'] = self.components
        self.phasemap.attrs['components_grid'] = [i+'_grid' for i in self.components]
        self.phasemap['mask'] = self.mask #should add grid dimensions automatically
        #must reset for serlialization to netcdf to work
        self.phasemap = self.phasemap.reset_index('grid').reset_coords(self.phasemap.attrs['components_grid'])

    def process_data(self,clean_params=None):
        
        self.phasemap['data'] = self.phasemap.raw_data.afl.scatt.clean(derivative=0,qlo=self.config['qlo'],qhi=self.config['qhi'])
        self.phasemap['deriv1'] = self.phasemap.raw_data.afl.scatt.clean(derivative=1,qlo=self.config['qlo'],qhi=self.config['qhi'])
        self.phasemap['deriv2'] = self.phasemap.raw_data.afl.scatt.clean(derivative=2,qlo=self.config['qlo'],qhi=self.config['qhi'])
        

    def label(self):
        self.update_status(f'Labelling data on iteration {self.iteration}')
        self.metric.calculate(self.phasemap)
        self.phasemap['W'] = (('sample_i','sample_j'),self.metric.W)
        self.phasemap.attrs['metric'] = str(self.metric.to_dict())
        
        self.labeler.label(self.phasemap)
        self.n_phases = self.labeler.n_phases

        ###XXX need to add cutonut for labelers that don't need silhouette or to use other methods
        ## In reality the determination of number of clusters should be handled in the PhaseLabeler object
        # self.n_cluster,labels,silh = PhaseLabeler.silhouette(self.metric.W,self.labeler)
        # self.n_cluster,labels,silh = PhaseLabeler.silhouette(self.metric.W,self.labeler)
        
        self.update_status(f'Found {self.labeler.n_phases} phases')

        self.phasemap.attrs['n_phases'] = self.n_phases
        self.phasemap['labels'] = ('sample',self.labeler.labels)
        self.phasemap = self.phasemap.afl.labels.make_ordinal()
        
    def extrapolate(self):
        # Predict phase behavior at each point in the phase diagram
        
        if self.n_phases==1:
            self.update_status(f'Using dummy GP for one phase')
            self.GP = GaussianProcess.DummyGP()
        else:
            self.update_status(f'Starting gaussian process calculation on {self.config["compute_device"]}')
            with tf.device(self.config['compute_device']):
                self.GP = GaussianProcess.GP(
                    self.phasemap,
                    self.components,
                    num_classes=self.n_phases
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
        self.acquisition.reset_phasemap(self.phasemap,self.components)
        self.acquisition.reset_mask(self.mask)
        self.phasemap = self.acquisition.calculate_metric(self.GP)

        self.update_status(f'Finding next sample composition based on acquisition function')
        check = self.data_manifest[['AL_mfrac_'+c for c in self.components]].values
        print(f"--> Skipping values: {check}")
        self.next_sample = self.acquisition.get_next_sample(composition_check=check)
        
        next_dict = self.next_sample.drop('grid').squeeze().to_pandas().to_dict()
        status_str = 'Next sample is '
        for k,v in next_dict.items():
            status_str += f'{k}: {v:4.3f} '
        self.update_status(status_str.strip())
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
        #self.phasemap['next_sample'] = ('component',self.next_sample.squeeze().values)
        #reset_index('grid').drop(['SLES3_grid','DEX_grid','CAPB_grid'])
        
        from xarray.core.merge import MergeError
        try:
            self.phasemap['next_sample'] = self.next_sample.drop('grid').squeeze()
        except MergeError:
            self.phasemap['next_sample'] = self.next_sample.drop('grid').squeeze().reset_coords(drop=True)
            
        self.phasemap.attrs['uuid'] = uuid_str
        self.phasemap.attrs['date'] = date
        self.phasemap.attrs['time'] = time
        self.phasemap.attrs['data_tag'] = self.config["data_tag"]
        self.phasemap.attrs['acq_count'] = self.acq_count
        self.phasemap.attrs['iteration'] = self.iteration
        self.phasemap.to_netcdf(save_path/f'phasemap_{self.config["data_tag"]}_{uuid_str}.nc')
        
        #write manifest csv
        AL_manifest_path = save_path/'calculation_manifest.csv'
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
        self.read_data()
        # self.process_data()
        self.label()
        self.extrapolate()
        self.get_next_sample()
        self.save_results()

    @Driver.unqueued(render_hint='precomposed_svg')
    def plot_scatt(self,**kwargs):
        if self.phasemap is None:
            return 'No phasemap loaded. Run read_data()'

        if 'labels_ordinal' not in self.phasemap:
            self.phasemap['labels_ordinal'] = ('system',np.zeros(self.phasemap.sizes['sample']))
            labels = [0]
        else:
            labels = np.unique(self.phasemap.labels_ordinal.values)
            
        if 'precomposed' in kwargs['render_hint']:
            matplotlib.use('Agg') #very important
    
            N = len(labels)
            fig,axes = plt.subplots(N,2,figsize=(8,N*4))

            if N==1:
                axes = np.array([axes])

            for i,label in enumerate(labels):
                spm = self.phasemap.set_index(sample='labels_ordinal').sel(sample=label)
                plt.sca(axes[i,0])
                spm.data.afl.scatt.plot_linlin(x='logq',legend=False);
            
                plt.sca(axes[i,1])
                spm.afl.comp.plot_discrete(components=self.phasemap.attrs['components']);
    
            svg  = mpl_plot_to_bytes(fig,format='svg')
            return svg
        elif kwargs['render_hint']=='raw': 
            # construct dict to send as json (all np.ndarrays must be converted to list!)
            
            out_dict = {}
            out_dict['components'] = self.phasemap.attrs['components']
            for i,label in enumerate(labels):
                out_dict[f'phase_{i}'] = {}
                
                spm = self.phasemap.set_index(sample='labels_ordinal').sel(sample=label)
                out_dict[f'phase_{i}']['labels'] = list(spm.labels.values)
                out_dict[f'phase_{i}']['labels_ordinal'] = int(label)
                out_dict[f'phase_{i}']['q'] = list(spm.q.values)
                #out_dict[f'phase_{i}']['raw_data'] = list(spm.raw_data.values)
                out_dict[f'phase_{i}']['data'] = list(spm.data.values)
                out_dict[f'phase_{i}']['compositions'] = {}
                for component in self.phasemap.attrs['components']:
                    out_dict[f'phase_{i}']['compositions'][component] = list(spm[component].values)
            return out_dict
        else:
            raise ValueError(f'Cannot handle render_hint={kwargs["render_hint"]}')

    @Driver.unqueued(render_hint='precomposed_svg')
    def plot_acq(self,**kwargs):
        if self.phasemap is None:
            return 'No phasemap loaded. Run read_data()'

        matplotlib.use('Agg') #very important
        fig,ax = plt.subplots()
        self.acquisition.plot()
        svg  = mpl_plot_to_bytes(fig,format='svg')
        return svg

    @Driver.unqueued(render_hint='precomposed_svg')
    def plot_gp(self,**kwargs):
        if self.phasemap is None:
            return 'No phasemap loaded. Run read_data()'

        if 'gp_y_mean' not in self.phasemap:
            raise ValueError('No GP results in phasemap. Run .predict()')

        if 'precomposed' in kwargs['render_hint']:
            matplotlib.use('Agg') #very important
            N = self.phasemap.sizes['phase_num']
            fig,axes = plt.subplots(self.phasemap.sizes['phase_num'],2,figsize=(8,N*4))
            if N==1:
                axes = np.array([axes])
            i = 0
            for (_,labels1),(_,labels2) in zip(self.phasemap.gp_y_mean.groupby('phase_num'),self.phasemap.gp_y_var.groupby('phase_num')):
                plt.sca(axes[i,0])
                self.phasemap.where(self.phasemap.mask).afl.comp.plot_continuous(components=self.phasemap.attrs['components_grid'],cmap='magma',labels=labels1.values);
                plt.sca(axes[i,1])
                self.phasemap.where(self.phasemap.mask).afl.comp.plot_continuous(components=self.phasemap.attrs['components_grid'],labels=labels2.values);
                i+=1

            img  = mpl_plot_to_bytes(fig,format=kwargs['render_hint'].split('_')[1])
            return img

        elif kwargs['render_hint']=='raw': 
            # construct dict to send as json (all np.ndarrays must be converted to list!)

            out_dict = {}
            out_dict['components'] = self.phasemap.attrs['components']
            out_dict['components_grid'] = self.phasemap.attrs['components_grid']
            for component in self.phasemap.attrs['components_grid']:
                out_dict[component] = list(self.phasemap[component].values)

            x,y = self.phasemap.afl.comp.to_xy(self.phasemap.attrs['components_grid']).T
            out_dict['x'] = list(x)
            out_dict['y'] = list(y)

            i =0
            for (_,labels1),(_,labels2) in zip(self.phasemap.gp_y_mean.groupby('phase_num'),self.phasemap.gp_y_var.groupby('phase_num')):
                out_dict[f'phase_{i}'] = {'mean':list(labels1.values),'var':list(labels2.values)}
                i+=1
            return out_dict
        else:
            raise ValueError(f'Cannot handle render_hint={kwargs["render_hint"]}')



    


   
