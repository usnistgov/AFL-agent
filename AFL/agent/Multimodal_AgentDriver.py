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
import warnings

import pandas as pd
import numpy as np
import xarray as xr
import pathlib
import tqdm

import h5py

import io
import matplotlib.pyplot as plt
import matplotlib

import AFL.agent.xarray_extensions
from AFL.agent import AcquisitionFunction 
from AFL.agent import GaussianProcess 
from AFL.agent import HschedGaussianProcess 
from AFL.agent import Metric 
from AFL.agent import PhaseLabeler
from AFL.agent.WatchDog import WatchDog


import tensorflow as tf

import gpflow

import uuid

class Multimodal_AgentDriver(Driver):
    defaults={}
    defaults['compute_device'] = '/device:CPU:0'
    defaults['data_path'] = '~/'
    defaults['AL_manifest_file'] = 'manifest.csv'
    defaults['save_path'] = '/home/AFL/'
    defaults['data_tag'] = 'default'
    defaults['subtract_background'] = False
    # defaults['grid_pts_per_row'] = 100
    def __init__(self,overrides=None):
        Driver.__init__(self,name='SAS_AgentDriver',defaults=self.gather_defaults(),overrides=overrides)

        self.watchdog = None 
        self.AL_manifest = None
        self._app = None
        self.name = 'SAS_AgentDriver'

        self.status_str = 'Fresh Server!'

        self.dataset = None
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
        status.append(f'Data Manifest:{self.config["AL_manifest_file"]}')
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
        
        path = pathlib.Path(self.config['AL_manifest_file'])
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
    
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self,value):
        self._mask=value
        self.masked_points = int(value.sum().values)
        self.total_points = value.size
    
    def subtract_background(self,path,fname):

        raw = self.read_csv(path,fname)
        raw.name = 'raw'

        empty_fname = 'MT-'+str(fname)
        empty = self.read_csv(path,empty_fname)
        empty.name = 'empty'

        sample_name = fname.replace('_chosen_r1d.csv','')
        h5_path = path / (sample_name+'.h5')
        with h5py.File(h5_path,'r') as h5:
            sample_transmission = h5['entry/sample/transmission'][()]

        empty_name = empty_fname.replace('_chosen_r1d.csv','')
        h5_path = path / (empty_name+'.h5')
        with h5py.File(h5_path,'r') as h5:
            empty_transmission = h5['entry/sample/transmission'][()]

        corrected = raw#/sample_transmission - empty.interp_like(raw)/empty_transmission
        corrected.name = 'corrected'
        return corrected,raw,empty,sample_transmission,empty_transmission

        
    def read_data(self):
        '''Read and process a dataset from a netcdf file'''
        self.update_status(f'Reading the latest data in {self.config["AL_manifest_file"]}')
        path = pathlib.Path(self.config['data_path'])
        
        self.dataset = xr.load_dataset(self.config['AL_manifest_file'])
        # look for multimodal_method, and AL_data+'_method'

        for AL_data in self.dataset.attrs['AL_data']:
            measurement = self.dataset[AL_data]
            kwargs = {
                xlo=self.dataset.attrs.get(AL_data+'_prep_xlo',None)
                xhi=self.dataset.attrs.get(AL_data+'_prep_xhi',None)
                xlo_isel=self.dataset.attrs.get(AL_data+'_prep_xlo_isel',None)
                xhi_isel=self.dataset.attrs.get(AL_data+'_prep_xhi_isel',None)
                pedestal=self.dataset.attrs.get(AL_data+'_prep_pedestal',None)
                npts=self.dataset.attrs.get(AL_data+'_prep_npts',None)
                sgf_window_length=self.dataset.attrs.get(AL_data+'__rep_sgf_window_length',2)
                sgf_polyorder=self.dataset.attrs.get(AL_data+'_prep_sgf_polyorder',2)
                logx=self.dataset.attrs.get(AL_data+'_prep_logx',False)
                logy=self.dataset.attrs.get(AL_data+'_prep_logy',False)
                }
            self.dataset[AL_data+'_deriv0'] = measurement.util.preprocess_data(derivative=0,**kwargs)
            self.dataset[AL_data+'_deriv1'] = measurement.util.preprocess_data(derivative=1,**kwargs)
            self.dataset[AL_data+'_deriv2'] = measurement.util.preprocess_data(derivative=2,**kwargs)
        
        self.dataset['data'] = self.dataset['data'] - self.dataset['data'].min('logq')
        
        if 'labels' not in self.dataset:
            self.dataset = self.dataset.afl.labels.make_default()

        self.components = self.dataset.attrs['components']

    def label(self):

        self.update_status(f'Labeling data on iteration {self.iteration}')

        self.metric.calculate(self.dataset)
        self.dataset['W'] = (('sample_i','sample_j'),self.metric.W)
        self.dataset.attrs['metric'] = str(self.metric.to_dict())
        
        self.labeler.label(self.dataset)
        self.n_phases = self.labeler.n_phases

        self.update_status(f'Found {self.labeler.n_phases} phases')

        self.dataset.attrs['_n_phases'] = self.n_phases
        self.dataset['labels'] = ('sample',self.labeler.labels)
        self.dataset = self.dataset.afl.labels.make_ordinal()
        
    def extrapolate(self):
        # Predict phase behavior at each point in the phase diagram
        
        if self.n_phases==1:
            self.update_status(f'Using dummy GP for one phase')
            self.classifier_GP = GaussianProcess.DummyGP(self.dataset)
        else:
            self.update_status(f'Starting gaussian process calculation on {self.config["compute_device"]}')
            with tf.device(self.config['compute_device']):
                kernel = gpflow.kernels.Matern32(variance=0.5,lengthscales=1.0) 
                self.classifier_GP = GaussianProcess.GP(
                    dataset = self.dataset,
                    kernel=kernel
                )
                self.classifier_GP.optimize(2000,progress_bar=True)


        for AL_data in self.dataset.attrs['AL_data']:
            if self.dataset.attrs[AL_data+'_AL_mode'] == 'regression':
        
            self.regression_GP = Hsched....
                
        self.acq_count   = 0
        self.iteration  += 1 #iteration represents number of full calculations

        self.update_status(f'Finished AL iteration {self.iteration}')
        
    @Driver.unqueued()
    def get_next_sample(self):
        self.update_status(f'Calculating acquisition function...')
        self.acquisition.reset_phasemap(self.dataset)
        self.dataset = self.acquisition.calculate_metric(self.GP)

        self.update_status(f'Finding next sample composition based on acquisition function')
        composition_check = self.dataset[self.dataset.attrs['components']]
        if 'sample' in composition_check.indexes:
            composition_check = composition_check.reset_index('sample').reset_coords(drop=True)
        composition_check = composition_check.to_array('component').transpose('sample',...)
        self.next_sample = self.acquisition.get_next_sample(
            composition_check = composition_check
        )
        
        next_dict = self.next_sample.squeeze().to_pandas().to_dict()
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
        self.dataset['gp_y_var'] = (('grid','phase_num'),self.acquisition.y_var)
        self.dataset['gp_y_mean'] = (('grid','phase_num'),self.acquisition.y_mean)
        #self.dataset['next_sample'] = ('component',self.next_sample.squeeze().values)
        #reset_index('grid').drop(['SLES3_grid','DEX_grid','CAPB_grid'])
        
        from xarray.core.merge import MergeError
        
        if 'component' in self.dataset.dims:
            warnings.warn('Dropping component dim (and all associated vars) from phasemap...')
            self.dataset = self.dataset.drop_dims('component')
            
        try:
            self.dataset['next_sample'] = self.acquisition.next_sample.squeeze()
        except MergeError:
            self.dataset['next_sample'] = self.acquisition.next_sample.squeeze().reset_coords(drop=True)
            
        self.dataset.attrs['uuid'] = uuid_str
        self.dataset.attrs['date'] = date
        self.dataset.attrs['time'] = time
        self.dataset.attrs['data_tag'] = self.config["data_tag"]
        self.dataset.attrs['acq_count'] = self.acq_count
        self.dataset.attrs['iteration'] = self.iteration
        self.dataset.to_netcdf(save_path/f'phasemap_{self.config["data_tag"]}_{uuid_str}.nc')
        
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

    def predict(self,datatype=None):
        self.read_data()
        self.label()
        self.extrapolate()
        self.get_next_sample()
        self.save_results()

    @Driver.unqueued(render_hint='precomposed_svg')
    def plot_scatt(self,**kwargs):
        if self.dataset is not None:
            if 'labels_ordinal' not in self.dataset:
                self.dataset['labels_ordinal'] = ('system',np.zeros(self.dataset.sizes['sample']))
                labels = [0]
            else:
                labels = np.unique(self.dataset.labels_ordinal.values)
            
        if 'precomposed' in kwargs['render_hint']:
            matplotlib.use('Agg') #very important
            if self.dataset is None:
                fig,ax = plt.subplots()
                plt.text(1,5,'No phasemap loaded. Run .read_data()')
                plt.gca().set(xlim=(0,10),ylim=(0,10))
            else:
                N = len(labels)
                fig,axes = plt.subplots(N,2,figsize=(8,N*4))

                if N==1:
                    axes = np.array([axes])

                for i,label in enumerate(labels):
                    spm = self.dataset.set_index(sample='labels_ordinal').sel(sample=label)
                    plt.sca(axes[i,0])
                    spm.data.afl.scatt.plot_linlin(x='logq',legend=False);
                
                    plt.sca(axes[i,1])
                    spm.afl.comp.plot_discrete(components=self.dataset.attrs['components']);
    
            svg  = mpl_plot_to_bytes(fig,format='svg')
            return svg
        elif kwargs['render_hint']=='raw': 
            # construct dict to send as json (all np.ndarrays must be converted to list!)
            out_dict = {}
            if self.dataset is None:
                out_dict = {'Error': 'No phasemap loaded. Run .read_data()'}
            else:
                out_dict['components'] = self.dataset.attrs['components']
                for i,label in enumerate(labels):
                    out_dict[f'phase_{i}'] = {}
                    
                    spm = self.dataset.set_index(sample='labels_ordinal').sel(sample=label)
                    out_dict[f'phase_{i}']['labels'] = list(spm.labels.values)
                    out_dict[f'phase_{i}']['labels_ordinal'] = int(label)
                    out_dict[f'phase_{i}']['q'] = list(spm.q.values)
                    #out_dict[f'phase_{i}']['raw_data'] = list(spm.raw_data.values)
                    out_dict[f'phase_{i}']['data'] = list(spm.data.values)
                    out_dict[f'phase_{i}']['compositions'] = {}
                    for component in self.dataset.attrs['components']:
                        out_dict[f'phase_{i}']['compositions'][component] = list(spm[component].values)
            return out_dict
        else:
            raise ValueError(f'Cannot handle render_hint={kwargs["render_hint"]}')

    @Driver.unqueued(render_hint='precomposed_svg')
    def plot_acq(self,**kwargs):
        matplotlib.use('Agg') #very important
        fig,ax = plt.subplots()
        if self.dataset is None:
            plt.text(1,5,'No phasemap loaded. Run .read_data()')
            plt.gca().set(xlim=(0,10),ylim=(0,10))
        else:
            self.acquisition.plot()
        svg  = mpl_plot_to_bytes(fig,format='svg')
        return svg

    @Driver.unqueued(render_hint='precomposed_svg')
    def plot_gp(self,**kwargs):
        if self.dataset is None:
            return 'No phasemap loaded. Run read_data()'

        if 'gp_y_mean' not in self.dataset:
            raise ValueError('No GP results in phasemap. Run .predict()')

        if 'precomposed' in kwargs['render_hint']:
            matplotlib.use('Agg') #very important
            N = self.dataset.sizes['phase_num']
            fig,axes = plt.subplots(self.dataset.sizes['phase_num'],2,figsize=(8,N*4))
            if N==1:
                axes = np.array([axes])
            i = 0
            for (_,labels1),(_,labels2) in zip(self.dataset.gp_y_mean.groupby('phase_num'),self.dataset.gp_y_var.groupby('phase_num')):
                plt.sca(axes[i,0])
                self.dataset.where(self.dataset.mask).afl.comp.plot_continuous(components=self.dataset.attrs['components_grid'],cmap='magma',labels=labels1.values);
                plt.sca(axes[i,1])
                self.dataset.where(self.dataset.mask).afl.comp.plot_continuous(components=self.dataset.attrs['components_grid'],labels=labels2.values);
                i+=1

            img  = mpl_plot_to_bytes(fig,format=kwargs['render_hint'].split('_')[1])
            return img

        elif kwargs['render_hint']=='raw': 
            # construct dict to send as json (all np.ndarrays must be converted to list!)

            out_dict = {}
            out_dict['components'] = self.dataset.attrs['components']
            out_dict['components_grid'] = self.dataset.attrs['components_grid']
            for component in self.dataset.attrs['components_grid']:
                out_dict[component] = list(self.dataset[component].values)

            x,y = self.dataset.afl.comp.to_xy(self.dataset.attrs['components_grid']).T
            out_dict['x'] = list(x)
            out_dict['y'] = list(y)

            i =0
            for (_,labels1),(_,labels2) in zip(self.dataset.gp_y_mean.groupby('phase_num'),self.dataset.gp_y_var.groupby('phase_num')):
                out_dict[f'phase_{i}'] = {'mean':list(labels1.values),'var':list(labels2.values)}
                i+=1
            return out_dict
        else:
            raise ValueError(f'Cannot handle render_hint={kwargs["render_hint"]}')



    


   
