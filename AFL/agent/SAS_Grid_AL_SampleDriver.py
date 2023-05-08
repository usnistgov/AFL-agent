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
    defaults['AL_manifest_file'] = '/home/afl642/USAXS_data/AL_manifest.nc'
    defaults['sample_manifest_file'] = '/home/afl642/USAXS_data/sample_manifest.nc'
    defaults['data_tag'] = 'default'
    def __init__(self,
            sas_url,
            agent_url,
            overrides=None, 
            data=None,
            ):

        Driver.__init__(self,name='_AL_SampleDriver',defaults=self.gather_defaults(),overrides=overrides,data=data)

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
        self.num_samples = 0
        self.blank_iter = None
        self.blank_well = None
        self.blank_plate_name = None
        
        self.loaded_plates = {}
        
    def status(self):
        status = []
        status.append(f'SAS: {self.sas_url}')
        status.append(self.status_str)
        status.append(self.AL_status_str)
        status.append(f'AL Components: {self.AL_components}')
        status.append(f'{self.num_samples} samples loaded')
        status.append(f'Measuring blank well {self.blank_well} on plate {self.blank_plate_name} every {self.blank_iter} iterations')
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
        self.AL_selection  = kwargs['AL_selection']
        pre_run_list       =  copy.deepcopy(kwargs.get('pre_run_list',[]))
        plate_to_slot      = kwargs['plate_to_slot']
        predict            = kwargs.get('predict',True)
        start_new_manifest            = kwargs.get('start_new_manifest',False)
        pts_per_row            = kwargs.get('pts_per_row',25)

        #handle blanks
        self.blank_iter            = kwargs.get('blank_iter',None)
        self.blank_well            = kwargs.get('blank_well',None)
        self.blank_plate_name      = kwargs.get('blank_plate_name',None)
        if self.blank_plate_name is not None:
            blank_plate_num = self.blank_plate_name.split('_')[1]
        if self.blank_well is not None:
            blank_row = self.blank_well[0]
            blank_col = int(self.blank_well[1:])


        # grab paths from config
        AL_manifest_path = pathlib.Path(self.config['AL_manifest_file'])
        sample_manifest_path = pathlib.Path(self.config['sample_manifest_file'])
        data_path = pathlib.Path(self.config['data_path'])
        data_manifest_path = pathlib.Path(self.config['data_manifest_file'])
        
        ################################
        ## LOAD MANIFEST & DOWNSELECT ##
        ################################
        self.sample_manifest = xr.load_dataset(sample_manifest_path)
        self.AL_selection['plate_name'] = list(plate_to_slot.keys())
        self.sample_manifest = mask_dataset(self.sample_manifest,self.AL_selection)
        self.num_samples = self.sample_manifest.sizes['sample']

        self.AL_components = self.sample_manifest.attrs['components']
        
        
        AL_iter = -1
        self.stop_AL = False
        while not self.stop_AL:
            AL_iter+=1

            self.app.logger.info(f'Starting new AL loop')

            #############################
            ## PERIODIC MEASURE BLANKS ##
            #############################
            if (AL_iter>0) and (self.blank_iter is not None) and (self.blank_well is not None) and (self.blank_plate_name is not None) and ((AL_iter%self.blank_iter)==0):
                sas_fname = f'p{blank_plate_num}_{self.blank_well}_y2_blank'
                blank_slot = plate_to_slot[self.blank_plate_name]
                self.update_status(f'Measuring blank well {self.blank_well} on {self.blank_plate_name} in slot {blank_slot}')
                self.sas_client.enqueue(task_name='setPosition',plate=blank_slot,row=blank_row,col=blank_col,y_offset=2)
                sas_uuid = self.sas_client.enqueue(task_name='expose',name=sas_fname,block=True)
                # self.sas_client.wait(sas_uuid)
            
            ###########################
            ## GET NEXT STEP FROM AL ##
            ###########################
            self.update_status('Getting next step from AL...')
            if pre_run_list:
                self.next_sample_AL = None
                next_sample_dict = pre_run_list.pop(0)
            else:
                #next_sample = self.agent_client.get_next_sample_queued()
                self.next_sample_AL = self.agent_client.get_object('next_sample')
                try:
                    next_sample_dict = self.next_sample_AL.squeeze().reset_coords('grid',drop=True).to_pandas().to_dict()
                except ValueError:
                    next_sample_dict = self.next_sample_AL.squeeze().to_pandas().to_dict()
                
            self.app.logger.info(f'Trying to measure next sample: {next_sample_dict}')
            
            ##############################
            ## FIND CLOSEST IN MANIFEST ##
            ##############################
            self.update_status('Finding closest sample in manifest...')
            
            # find closest in manifest
            coords_available = self.sample_manifest[self.AL_components].to_array('component')
            ds_next_sample = pd.Series(next_sample_dict).to_frame().T.to_xarray().rename_dims(index='sample').reset_coords()
            coords_new = ds_next_sample[self.AL_components].to_array('component').transpose(...,'component')
    
            coords_available = self.sample_manifest[self.AL_components].to_array('component').transpose(...,'component')
            sample_distances = cdist(coords_new,coords_available)
            self.next_sample = self.sample_manifest.isel(sample=sample_distances.argmin()).reset_coords()
            self.sample_manifest = self.sample_manifest.drop_isel(sample=sample_distances.argmin())#remove entry from sample_list
            self.num_samples = self.sample_manifest.sizes['sample']#update num samples
            
            ###############################
            ## MEASURE OR READ FROM DISK ##
            ###############################
            # check if already measured
            next_plate_name = self.next_sample.plate_name.values[()]
            next_plate_num = int(self.next_sample.plate.values[()])
            _,next_well = parse_well(self.next_sample.dest.values[()])
            next_well_row = next_well[0]
            next_well_col = int(next_well[1:])
            sas_fname = f'p{next_plate_num}_{next_well}_y2'
            slot_name = plate_to_slot[next_plate_name]

            # reload data manifest and check if already measured
            self.df_data_manifest = pd.read_csv(data_manifest_path)
            try:
                sel = self.df_data_manifest.set_index(['plate','well']).loc[next_plate_num,next_well]
            except KeyError:
                sel = None

            
            if sel is None:
                self.update_status(f'Measuring well {next_well} on plate {next_plate_name} in slot {slot_name}')
                self.sas_client.enqueue(task_name='setPosition',plate=slot_name,row=next_well_row,col=next_well_col,y_offset=2)
                sas_uuid = self.sas_client.enqueue(task_name='expose',name=sas_fname,block=True)
                self.sas_client.wait(sas_uuid)

                sleep_time=5
                for i in range(25):
                    print(f'--> Iteration {i} on trying to find sas data...waited {i*sleep_time} seconds...')
                    sas_fpath = list(data_path.glob(f'{sas_fname}*.h5'))
                    sas_fpath = sorted(sas_fpath,key = lambda x: str(x).split('_')[-1])
                    if len(sas_fpath)==0:
                        time.sleep(sleep_time)
                    else:
                        break

                if len(sas_fpath)==0:
                    raise ValueError('No file found after measurement...')
                else:
                    sas_fpath = sas_fpath[-1]


                # add to manifest
                row = {}
                row['well'] = next_well
                row['plate'] = next_plate_num
                row['plate_name'] = next_plate_name
                row['sample_name'] = self.next_sample.sample_name.values[()]
                row['sample'] = self.df_data_manifest.shape[0]
                row['MOF'] = self.next_sample.MOF.values[()]
                row['fpath'] = sas_fpath
                self.df_data_manifest  = self.df_data_manifest.append(row,ignore_index=True)
                self.df_data_manifest.to_csv(data_manifest_path)

            else:
                sas_fpath = sel.fpath
                sas_fpath = sorted(sel.fpath.values,key = lambda x: str(x).split('_')[-1])[-1]
                self.update_status(f'Reading data for well {next_well} on plate {next_plate_name} in slot {slot_name}: {sas_fpath}')
                
                
            #read data and add to next_sample dict
            self.new_data = self.next_sample.copy()
            with h5py.File(sas_fpath,'r') as h5:
                reduced = reduce_uascan(h5)

            da_sas = xr.DataArray(reduced['R'],coords={'q':reduced['Q']},dims=['q'])
            da_sas['q'] = da_sas.q.pipe(np.abs)
            da_sas = da_sas.groupby('q').mean()
            da_sas = da_sas.interp(q=np.geomspace(1e-5,1e-1,500))
            self.new_data['R'] = da_sas
            self.new_data.attrs['AL_data'] = 'R'
            
            ########################
            ## UPDATE AL MANIFEST ##
            ########################
            if start_new_manifest and AL_manifest_path.exists():
                self.update_status(f'Backing up old and then starting new manifest...')
                AL_manifest_backup = xr.load_dataset(AL_manifest_path)
                i=0
                while True:
                    AL_manifest_backup_path = pathlib.Path(str(AL_manifest_path)+f'.bak{i}')
                    if AL_manifest_backup_path.exists():
                        i+=1
                    else:
                        print(f'--> Writing backup AL file to {AL_manifest_backup_path}')
                        AL_manifest_backup.to_netcdf(AL_manifest_backup_path)
                        break

                self.AL_manifest = self.new_data.expand_dims('sample')
                start_new_manifest=False

            elif AL_manifest_path.exists():
                self.update_status(f'Updating manifest...')
                self.AL_manifest = xr.load_dataset(AL_manifest_path)

                if 'grid' in self.AL_manifest.dims:
                    self.AL_manifest = self.AL_manifest.drop_dims('grid')

                self.AL_manifest = xr.concat([self.AL_manifest,self.new_data],dim='sample')

            else:
                self.update_status(f'Starting new manifest...')
                self.AL_manifest = self.new_data.expand_dims('sample')
            
            self.AL_manifest.attrs.update(self.sample_manifest.attrs)
            self.AL_manifest.attrs['AL_data'] = 'R'

            self.AL_manifest  = self.AL_manifest.afl.comp.add_grid(pts_per_row=pts_per_row)

            # self.AL_manifest.attrs['components'] = self.AL_components
            # self.AL_manifest.attrs['GP_domain_transform'] = 'range_scaled'
            #  for component in self.AL_components:
            #      self.AL_manifest.attrs[f'{component}_range'] = self.AL_component_ranges[component]
            #      self.AL_manifest.attrs[f'{component}_grid_range'] = self.AL_component_ranges[component]
            self.AL_manifest.to_netcdf(AL_manifest_path)

            ################
            ## TRIGGER AL ##
            ################
            self.update_status(f'Triggering agent server...')
            if len(pre_run_list)==0:
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
