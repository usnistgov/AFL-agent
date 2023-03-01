from AFL.automation.APIServer.Client import Client
from AFL.automation.prepare.OT2Client import OT2Client
from AFL.automation.shared.utilities import listify
from AFL.automation.APIServer.Driver import Driver
from AFL.agent.AgentClient import AgentClient
from AFL.automation.shared.units import units

from AFL.agent.reduce_usaxs import reduce_uascan

from scipy.spatial.distance import cdist

import warnings

import pandas as pd
import numpy as np 
import xarray as xr 
from math import ceil,sqrt
import json
import time
import requests
import shutil
import datetime
import traceback
import pathlib
import uuid
import copy

import AFL.automation.prepare
import shutil
import h5py


class SAS_Grid_AL_SampleDriver(Driver):
    defaults={}
    defaults['data_path'] = '/mnt/aux/chess/reduced_data/cycles/2022-1/id3b/beaucage-2324-D/analysis/'
    defaults['data_manifest_file'] = '/home/afl642/USAXS_data/usaxs_manifest.csv'
    defaults['sample_manifest_file'] = '/home/afl642/USAXS_data/sample_manifest.nc'
    defaults['AL_manifest_file'] = '/home/afl642/USAXS_data/AL_manifest.nc'
    defaults['data_tag'] = 'default'
    def __init__(self,
            sas_url,
            agent_url,
            overrides=None, 
            ):

        Driver.__init__(self,name='_AL_SampleDriver',defaults=self.gather_defaults(),overrides=overrides)

        if not (len(agent_url.split(':'))==2):
            raise ArgumentError('Need to specify both ip and port on agent_url')

        self.app = None
        self.name = 'SAS_Grid_AL_SampleDriver'

        #measure samples
        self.sas_url = sas_url
        self.sas_client = Client(sas_url.split(':')[0],port=sas_url.split(':')[1])
        self.sas_client.login('SampleServer_SASClient')
        self.sas_client.debug(False)
        
        #agent
        self.agent_url = agent_url
        self.agent_client = AgentClient(agent_url.split(':')[0],port=agent_url.split(':')[1])
        self.agent_client.login('SampleServer_AgentClient')
        self.agent_client.debug(False)

        self.status_str = 'Fresh Server!'
        
        self.catch_protocol=None
        self.AL_status_str = ''
        self.sample_manifest  = None
        self.data_manifest  = None
        self.AL_manifest  = None
        self.AL_components = None
        self.components = None
        
        self.loaded_plates = {}
        
    def status(self):
        status = []
        status.append(f'SAS: {self.sas_url}')
        status.append(self.status_str)
        status.append(self.AL_status_str)
        status.append(f'AL Components: {self.AL_components}')
        status.append(f'{self.num_samples} samples loaded')
        return status
    
    def update_status(self,value):
        self.status_str = value
        self.app.logger.info(value)

    def measure(self,sample):
        exposure = sample.get('exposure',None)

        sas_uuid = self.sas_client.enqueue(task_name='expose',name=sample['name'],block=True,exposure=exposure)
        self.sas_client.wait(sas_uuid)

        return sas_uuid

    def sample(self,sample):
        return self.process_sample(sample)
    
    @Driver.unqueued()
    def stop_active_learning(self):
        self.stop_AL = True
        
        
    def active_learning_loop(self,**kwargs):
        self.components    = kwargs['components']
        self.AL_components = kwargs['AL_components']
        self.AL_component_ranges = kwargs['AL_component_ranges']
        self.AL_selection  = kwargs['AL_selection']
        pre_run_list       =  copy.deepcopy(kwargs.get('pre_run_list',[]))
        exposure           = kwargs['exposure']
        empty_exposure     = kwargs['empty_exposure']
        predict            = kwargs.get('predict',True)
        plate_to_slot      = kwargs['plate_to_slot']


        # grab paths from config
        AL_manifest_path = pathlib.Path(self.config['AL_manifest_file'])
        sample_manifest_path = pathlib.Path(self.config['sample_manifest_file'])
        data_path = pathlib.Path(self.config['data_path'])
        data_manifest_path = pathlib.Path(self.config['data_manifest_file'])
        
        # load sample manifest and downselect
        self.sample_manifest = xr.load_dataset(sample_manifest_path)
        self.AL_selection['plate_name'] = list(self.loaded_plates.keys())
        self.sample_manifest = self.mask_dataset(self.sample_manifest,self.AL_selection)
        self.num_samples = self.sample_manifest.sizes['sample']
        
        
        self.stop_AL = False
        while not self.stop_AL:
            self.app.logger.info(f'Starting new AL loop')
            
            ###########################
            ## GET NEXT STEP FROM AL ##
            ###########################
            self.update_status('Getting next step from AL...')
            if pre_run_list:
                self.next_sample = None
                next_sample_dict = pre_run_list.pop(0)
            else:
                #next_sample = self.agent_client.get_next_sample_queued()
                self.next_sample = self.agent_client.get_object('next_sample')
                next_sample_dict = self.next_sample.squeeze().reset_coords('grid',drop=True).to_pandas().to_dict()
                
            self.app.logger.info(f'Trying to measure next sample: {next_sample_dict}')
            
            ##############################
            ## FIND CLOSEST IN MANIFEST ##
            ##############################
            self.update_status('Finding closest sample in manifest...')
            
            # find closest in manifest
            coords_available = self.sample_manifest[self.AL_components].to_array('component')
            ds_next_sample = pd.DataFrame(next_sample_dict).to_xarray().rename_dims(index='sample')
            coords_new = ds_next_sample[self.AL_components].to_array('component').transpose(...,'component')
    
            coords_available = self.sample_manifest[AL_components].to_array('component').transpose(...,'component')
            sample_distances = cdist(coords_new,coords_available)
            next_sample = self.sample_manifest.isel(sample=sample_distances.argmin())
            
            ###############################
            ## MEASURE OR READ FROM DISK ##
            ###############################
            # check if already measured
            next_plate_name = next_sample.plate.values[()]
            _,next_well = parse_well(next_sample.dest.values[()])
            next_well_row = next_well[0]
            next_well_col = next_well[1:]
            sas_fname = 'p{next_plate_name}-{next_well}-y2'
            slot_name = 

            # reload data manifest and check if already measured
            df_data_manifest = pd.read_csv(data_manifest_path)
            try:
                sel = df_data_manifest.set_index(['plate','well']).loc[next_plate_name,next_well]
            except KeyError:
                sel = None

            
            if sel is None:
                raise ValueError('Not testing this yet....')
                slot_name = plate_to_slot[next_plate_name]
                self.update_status(f'Measuring well {next_well} on plate {next_plate_name} in slot {slot_name}')
                sas_client.enqueue(task_name='setPosition',plate=slot_name,row=next_well_row,col=next_well_col,y_offset=2)
                sas_client.enqueue(task_name='expose',name=sas_fname,block=True)

                for i in range(25):
                    sas_fpath = list(data_path.glob(f'{sas_fname}*.h5'))
                    sas_fpath = sorted(sas_fpath,key = lambda x: str(x).split('_')[-1])
                    if len(sas_fpath)==0:
                        time.sleep(5)
                    else:
                        break

                if len(sas_fpath)==0:
                    raise ValueError('No file found after measurement...')
                else:
                    sas_fpath = sas_fpath([-1])


                # add to manifest
                row = {}
                row['well'] = next_well
                row['plate'] = next_plate_name
                row['sample_name'] = next_sample.sample_name.values[()]
                row['MOF'] = next_sample.MOF.values[()]
                row['fpath'] = sas_fpath
                df_data_manifest.append(row,ignore_index=True)
                df_data_manifest.to_csv(data_manifest_path)

            else:
                self.update_status(f'Reading data for well {next_well} on plate {next_plate_name} in slot {slot_name}')
                sas_fpath = sel.fpath
                
                
            #read data and add to next_sample dict
            new_data = next_sample.copy()
            with h5py.File(sas_fpath,'r') as h5:
                reduced = reduce_uascan(h5)
                da = xr.DataArray(reduces['R'],coords={'q':reduced['Q']},dims=['q'])
                da['q'] = da.q.pipe(np.abs)
                da = da.groupby('q').mean()
                da = da.interp(q=np.geomspace(1e-5,1e-1,500))


            new_data['R'] = sas_data
            new_data.attrs['AL_data'] = 'R'
            
            ########################
            ## UPDATE AL MANIFEST ##
            ########################
            self.update_status(f'Updating manifest...')
            if AL_manifest_path.exists():
                self.AL_manifest = xr.read_dataset(AL_manifest_path)
                self.AL_manifest = xr.concat([self.AL_manifest,new_data],dim='sample')
            else:
                self.AL_manifest = new_data
            
            self.AL_manifest.attrs['AL_data'] = 'R'
            self.AL_manifest.attrs['components'] = self.AL_components
            self.AL_manifest.attrs['GP_domain_transform'] = 'range_scaled'
            for component in self.AL_components:
                self.AL_manifest.attrs[f'{component}_range'] = self.AL_component_ranges[component]
                self.AL_manifest.attrs[f'{component}_grid_range'] = self.AL_component_ranges[component]
            self.AL_manifest.to_netcdf(AL_manifest_path)

            ################
            ## TRIGGER AL ##
            ################
            self.update_status(f'Triggering agent server...')
            if predict:
                self.agent_uuid = self.agent_client.enqueue(task_name='predict',datatype='nc')
            
                # wait for AL
                self.app.logger.info(f'Waiting for agent...')
                self.agent_client.wait(self.agent_uuid)
            else:#used for intialization
                return
            
            
   

def mask_dataset(dataset,sel_dict): 
    masks_and = []
    for var,values in sel_dict.items():
        masks_or = []
        for value in values:
            masks_or.append(dataset[var]==value)
        masks_and.append(np.logical_or.reduce(masks_or))
    mask = np.logical_and.reduce(masks_and)
    mask = xr.ones_like(dataset[var]).copy(data=mask)#just use last var in list...hopefully this isn't folly
    return dataset.copy().where(mask,drop=True)


   

def parse_well(loc):
    for i,loc_part in enumerate(list(loc)):
        if loc_part.isalpha():
            break
    slot = loc[:i]
    well = loc[i:]
    return slot,well
