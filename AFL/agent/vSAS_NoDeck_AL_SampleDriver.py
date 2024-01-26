from AFL.automation.APIServer.Client import Client
from AFL.automation.prepare.OT2Client import OT2Client
from AFL.automation.shared.utilities import listify
from AFL.automation.APIServer.Driver import Driver
from AFL.agent.AgentClient import AgentClient
from AFL.automation.shared.units import units
from AFL.automation.prepare.Sample import Sample

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
import os

import AFL.automation.prepare
import shutil
import h5py

from tiled.client import from_uri
from tiled.queries import Eq,Contains


class vSAS_NoDeck_AL_SampleDriver(Driver):
    defaults={}
    defaults['snapshot_directory'] = '/home/nistoroboto'
    defaults['csv_data_path'] = '/nfs/aux/chess/reduced_data/cycles/2022-1/id3b/beaucage-2324-D/analysis/'
    defaults['AL_manifest_file'] = '/nfs/aux/chess/reduced_data/cycles/2022-1/id3b/beaucage-2324-D/analysis/data_manifest.csv'
    defaults['data_tag'] = 'default'
    defaults['max_sample_transmission'] = 0.6
    def __init__(self,
            load_url,
            prep_url,
            sas_url,
            agent_url,
            tiled_url,
            spec_url=None,
            camera_urls = None,
            snapshot_directory =None,
            overrides=None, 
            ):

        Driver.__init__(self,name='vSAS_NoDeck_AL_SampleDriver',defaults=self.gather_defaults(),overrides=overrides)

        if not (len(load_url.split(':'))==2):
            raise ArgumentError('Need to specify both ip and port on load_url')

        if not (len(prep_url.split(':'))==2):
            raise ArgumentError('Need to specify both ip and port on prep_url')

        if not (len(agent_url.split(':'))==2):
            raise ArgumentError('Need to specify both ip and port on agent_url')

        self.app = None
        self.name = 'vSAS_AL_SampleDriver'

        #prepare samples
        # self.prep_url = prep_url
        # self.prep_client = OT2Client(prep_url.split(':')[0],port=prep_url.split(':')[1])
        # self.prep_client.login('SampleServer_PrepClient')
        # self.prep_client.debug(False)

        #load samples
        # self.load_client = Client(load_url.split(':')[0],port=load_url.split(':')[1])
        # self.load_client.login('SampleServer_LoadClient')
        # self.load_client.debug(False)
 
        #measure samples
        self.sas_url = sas_url
        self.sas_client = Client(sas_url.split(':')[0],port=sas_url.split(':')[1])
        self.sas_client.login('SampleServer_SASClient')
        self.sas_client.debug(False)
        
        # #vSAS additional stuff
        # netcdf_path = '/Users/drs18/Documents/multimodal-dev/phasemap_P188_2D_MultiModal_UCB_noThomp_FixedP188_30524b88-00f7-4606-9c4d-57ad7880f95e.nc'
        # model_ds = xr.load_dataset(netcdf_path)
        # self.sas_client.set_driver_object(dataset=model_ds)
        # self.sas_client.set_driver_object(clustered=True)
        # self.sas_client.enqueue(task_name='load_model_dataset')
        # self.sas_client.enqueue(task_name='generate_model')
        # self.sas_client.enqueue(task_name='train_model',niter=1001)
        
        
        #agent
        self.agent_url = agent_url
        self.agent_client = AgentClient(agent_url.split(':')[0],port=agent_url.split(':')[1])
        self.agent_client.login('SampleServer_AgentClient')
        self.agent_client.debug(False)

        if spec_url is not None:
            self.spec_url = spec_url
            self.spec_client = Client(spec_url.split(':')[0],port=spec_url.split(':')[1])
            self.spec_client.login('SampleServer_LSClient')
            self.spec_client.debug(False)
            self.spec_client.enqueue(task_name='setExposure',time=0.1)
        else:
            self.spec_client = None

        # start tiled catalog connection
        self.tiled_cat = from_uri(tiled_url,api_key=os.environ['TILED_API_KEY'])
    
        if camera_urls is None:
            self.camera_urls = []
        else:
            self.camera_urls = camera_urls

        if snapshot_directory is not None:
            self.config['snapshot_directory'] = snapshot_directory

        self.rinse_uuid = None
        self.prep_uuid = None
        self.catch_uuid = None
        self.agent_uuid = None

        self.status_str = 'Fresh Server!'
        self.wait_time = 30.0 #seconds
        
        
        self.catch_protocol=None
        self.AL_status_str = ''
        self.data_manifest  = None
        self.AL_components = None
        self.components = None


       
    def status(self):
        status = []
        status.append(f'Snapshots: {self.config["snapshot_directory"]}')
        status.append(f'Cameras: {self.camera_urls}')
        status.append(f'SAS: {self.sas_url}')
        status.append(f'Sample Wait Time: {self.wait_time}')
        status.append(f'AL_components: {self.AL_components}')
        status.append(self.status_str)
        status.append(self.AL_status_str)
        status.append(f'Components: {self.components}')
        status.append(f'AL Components: {self.AL_components}')
        return status
    
    def update_status(self,value):
        self.status_str = value
        self.app.logger.info(value)

    def measure(self,sample):
        exposure = sample.get('exposure',None)

        if self.spec_client is not None:
            self.spec_client.set_config(
                    filepath='/home/pi/2305_SINQ_data_reduced/',
                    filename=f'{sample["name"]}-beforeSAS-spec.h5'
                )
            self.spec_client.enqueue(task_name='collectContinuous',duration=5,interactive=False)

        ### the point in which the virtual SAS data driver will produce a spectrum and store it to tiled
        sas_uuid = self.sas_client.enqueue(task_name='expose',name=sample['name'],block=True,exposure=exposure)
        self.sas_client.wait(sas_uuid)

        if self.spec_client is not None:
            self.spec_client.set_config(
                    filepath='/home/pi/2305_SINQ_data_reduced/',
                    filename=f'{sample["name"]}-afterSAS-spec.h5'
                )
            spec_uuid = self.spec_client.enqueue(task_name='collectContinuous',duration=5,interactive=False)
            self.spec_client.wait(spec_uuid)

        return sas_uuid

    def sample(self,sample):
        return self.process_sample(sample)
        
    def process_sample(self,sample):
        name = sample['name']

        targets = set()
        for task in sample['prep_protocol']:
            if 'target' in task['source'].lower():
                targets.add(task['source'])
            if 'target' in task['dest'].lower():
                targets.add(task['dest'])

        for task in sample['catch_protocol']:
            if 'target' in task['source'].lower():
                targets.add(task['source'])
            if 'target' in task['dest'].lower():
                targets.add(task['dest'])

        target_map = {}
        for t in targets:
            prep_target = self.prep_client.enqueue(task_name='get_prep_target',interactive=True)['return_val']
            target_map[t] = prep_target

        for i,task in enumerate(sample['prep_protocol']):
            #if the well isn't in the map, just use the well
            task['source'] = target_map.get(task['source'],task['source'])
            task['dest'] = target_map.get(task['dest'],task['dest'])
            if i==0:
                task['force_new_tip']=True
            if i==(len(sample['prep_protocol'])-1):#last prepare
                task['drop_tip']=False
            self.prep_uuid = self.prep_client.transfer(**task)
 
        if self.rinse_uuid is not None:
            self.update_status(f'Waiting for rinse...')
            self.load_client.wait(self.rinse_uuid)
            self.update_status(f'Rinse done!')

        if self.spec_client is not None:
            self.spec_client.set_config(
                    filepath='/home/pi/2305_SINQ_data_reduced/',
                    filename=f'{sample["name"]}-MT-spec.h5'
                )
            self.spec_client.enqueue(task_name='collectContinuous',duration=5,interactive=False)

        if sample['empty_exposure']>0.0:
            self.update_status(f'Cell is clean, measuring empty cell scattering...')
            empty = {}
            empty['name'] = 'MT-'+sample['name']
            empty['exposure'] = sample['empty_exposure']
            empty['wiggle'] = False
            self.sas_uuid = self.measure(empty)
            self.sas_client.wait(self.sas_uuid)
                
        if self.prep_uuid is not None: 
            self.prep_client.wait(self.prep_uuid)
            self.take_snapshot(prefix = f'02-after-prep-{name}')

        
        self.update_status(f'Queueing sample {name} load into syringe loader')
        for task in sample['catch_protocol']:
            #if the well isn't in the map, just use the well
            task['source'] = target_map.get(task['source'],task['source'])
            task['dest'] = target_map.get(task['dest'],task['dest'])
            self.catch_uuid = self.prep_client.transfer(**task)
        
        if self.catch_uuid is not None:
            self.update_status(f'Waiting for sample prep/catch of {name} to finish: {self.catch_uuid}')
            self.prep_client.wait(self.catch_uuid)
            self.take_snapshot(prefix = f'03-after-catch-{name}')

        #homing robot to try to mitigate drift problems
        self.prep_client.home()

        if self.spec_client is not None:
            self.spec_client.set_config(
                    filepath='/home/pi/2305_SINQ_data_reduced/',
                    filename=f'{sample["name"]}-duringLoad-spec.h5'
                )
            self.spec_client.enqueue(task_name='collectContinuous',duration=15,interactive=False)
        
        self.load_uuid = self.load_client.enqueue(task_name='loadSample',sampleVolume=sample['volume'])
        self.update_status(f'Loading sample into cell: {self.load_uuid}')
        self.load_client.wait(self.load_uuid)
        self.take_snapshot(prefix = f'05-after-load-{name}')
        
        self.update_status(f'Sample is loaded, asking the instrument for exposure...')
        self.sas_uuid = self.measure(sample)

        self.update_status(f'Cleaning up sample {name}...')
        self.rinse_uuid = self.load_client.enqueue(task_name='rinseCell')

        self.update_status(f'Waiting for instrument to measure scattering of {name} with UUID {self.sas_uuid}...')
        self.sas_client.wait(self.sas_uuid)
        self.take_snapshot(prefix = f'06-after-measure-{name}')
            
        self.update_status(f'All done for {name}!')
        
    def mfrac_to_mass(self,mass_fractions,specified_conc,sample_volume,output_units='mg'):
        if not (len(mass_fractions)==3):
            raise ValueError('Only ternaries are currently supported. Need to pass three mass fractions')
            
        if len(specified_conc)>1:
            raise ValueError('Only one concentration should be fixed!')
        specified_component = list(specified_conc.keys())[0]
        
        components = list(mass_fractions.keys())
        components.remove(specified_component)
        
        xB = mass_fractions[components[0]]*units('')
        xC = mass_fractions[components[1]]*units('')
        XB = xB/(1-xB)
        XC = xC/(1-xC)
        
        mA = (specified_conc[specified_component]*sample_volume)
        mC = mA*(XC+XB*XC)/(1-XB*XC)
        mB = XB*(mA+mC)
        
        mass_dict = {}
        mass_dict[specified_component] = (mA).to(output_units)
        mass_dict[components[0]] = (mB).to(output_units)
        mass_dict[components[1]] = (mC).to(output_units)
        return mass_dict


    @Driver.unqueued()
    def stop_active_learning(self):
        self.stop_AL = True
    
    def take_snapshot(self,*args,**kwargs):
        pass
    
    def active_learning_loop(self,**kwargs):
        # grid must have units for each component specified, hopefully this will progate to next_sample....
        self.components     = kwargs['components']
        self.AL_components  = kwargs['AL_components']
        self.AL_kwargs      = kwargs.get('AL_kwargs',{})
        self.fixed_concs    = kwargs.get('fixed_concs',{})
        predict             = kwargs.get('predict',True)
        start_new_manifest  = kwargs.get('start_new_manifest',False)
        ternary             = kwargs['ternary']#should be bool
        pre_run_list        = copy.deepcopy(kwargs.get('pre_run_list',[]))

        AL_manifest_path = pathlib.Path(self.config['AL_manifest_file'])
        data_path = pathlib.Path(self.config['csv_data_path'])
        
        self.stop_AL = False
        while not self.stop_AL:
            self.app.logger.info(f'Starting new AL loop')
            #####################
            ## GET NEXT SAMPLE ##
            #####################
            if pre_run_list:
                self.next_sample = None
                next_sample_dict = pre_run_list.pop(0)
            else:
                self.next_sample = self.agent_client.get_object('next_sample')
                next_sample_dict = self.next_sample.squeeze()
                if 'grid' in next_sample_dict:
                    next_sample_dict = next_sample_dict.reset_coords('grid',drop=True)
                next_sample_dict = next_sample_dict.to_pandas().to_dict()
                next_sample_dict = {k:{'value':v,'units':self.next_sample.attrs[k+'_units']} for k,v in next_sample_dict.items()}
            self.app.logger.info(f'Preparing to make next sample: {next_sample_dict}')
                    
            
            ##############################
            ## PROTOCOL FOR NEXT SAMPLE ##
            ##############################
            if ternary and (len(self.components)==4):
                conc_spec = {}
                for name,value in self.fixed_concs.items():
                    conc_spec[name] = value['value']*units(value['units'])
                
                mass_dict = self.mfrac_to_mass(
                    mass_fractions=next_sample_dict,
                    specified_conc=conc_spec,
                    sample_volume=1*units("ml"),
                    output_units='mg')
            elif ternary and (len(self.components)==3):
                mass_dict = {}
                sample_mass = 1*units('mg')
                for name,comp in next_sample_dict.items():
                    mass_dict[name] = (comp['value']*units(comp['units'])*sample_mass).to('mg')
                
            else:
                #assume concs for now...
                if len(next_sample_dict)<(len(self.components)-1):
                    raise ValueError('System under specified...')

                mass_dict = {}
                for name,comp in next_sample_dict.items():
                    mass_dict[name] = (comp['value']*units(comp['units'])*sample_volume).to('mg')
            
            self.target = AFL.automation.prepare.Solution('target',self.components)
            for k,v in mass_dict.items():
                self.target[k].mass = v
            self.sample = Sample('dummy',self.target,target_check=self.target,balancer=None)
            validated=True

            #######################################
            ## INIITLIAZE DATASET FOR NEW SAMPLE ##
            #######################################
            sample_uuid = str(uuid.uuid4())
            sample_name = f'AL_{self.config["data_tag"]}_{sample_uuid[-8:]}'

            self.new_data              = xr.Dataset()
            self.new_data['sample_name']     = sample_name
            self.new_data['label']     = -1
            self.new_data['validated'] = validated
            self.new_data['sample_uuid']      = sample_uuid
            sample_composition = {}
            if ternary:
                total = 0
                for component in self.AL_components:
                    mf = self.sample.target_check.mass_fraction[component].magnitude
                    self.new_data[component] = mf
                    total+=mf
                for component in self.AL_components:
                    self.new_data[component] = self.new_data[component]/total
                    
                    sample_composition[component] = {}
                    sample_composition[component]['value'] = self.new_data[component].values[()]
                    sample_composition[component]['units'] = 'fraction'
            else:
                for component in self.AL_components:
                    self.new_data[component] = self.sample.target_check.concentration[component].to("mg/ml").magnitude
                    self.new_data[component].attrs['units'] = 'mg/ml'
                    
                    sample_composition[component] = {}
                    sample_composition[component]['value'] = self.new_data[component].values[0]
                    sample_composition[component]['units'] = 'mg/ml'
                    
                
            for component in self.components:
                self.new_data['mfrac_'+component] = self.sample.target_check.mass_fraction[component].magnitude
                self.new_data['mass_'+component] = self.sample.target_check[component].mass.to('mg').magnitude
                self.new_data['mass_'+component].attrs['units'] = 'mg'
                    
            
            #################
            ##  SET SAMPLE ##
            #################
            sample_data = self.set_sample(sample_name=sample_name,sample_uuid=sample_uuid, sample_composition=sample_composition)
            print(sample_data)
            #self.prep_client.enqueue(task_name='set_sample',**sample_data)
            #self.load_client.enqueue(task_name='set_sample',**sample_data)
            self.sas_client.enqueue(task_name='set_sample',**sample_data)
            self.agent_client.enqueue(task_name='set_sample',**sample_data)
            # if self.spec_client is not None:
            #     self.spec_client.enqueue(task_name='set_sample',**sample_data)
                
            ####################
            ## PROCESS SAMPLE ##
            ####################
            #self.process_sample(
            self.measure(
                    dict(
                        name=sample_name,
                        prep_protocol = [],
                        catch_protocol =[],
                        volume = 1.0,
                        exposure = 1.0,
                        empty_exposure = 0.0
                )
            )
            
            ########################
            ## PROCESS MESUREMENT ##
            ########################
            res = self.tiled_cat.search(Eq('sample_uuid',sample_uuid)).search(Eq('task_name','expose'))
            if len(res)==0:
                self.app.logger.info(f'No expose task found for {sample_uuid}')
                self.AL_status_str = 'No expose task found'
                raise ValueError('No expose task found')
            array_client = list(res.items())[-1][-1]#first [-1] is for last expose, second is for actual ArrayClient rather than key
            self.new_data['SAS'] = xr.DataArray(
                data=array_client.metadata['I'],
                dims=['q'],
                coords={'q':array_client.metadata['q']}
            )
            
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
            
            self.AL_manifest.attrs.update(self.AL_kwargs)
            self.AL_manifest.attrs['AL_data'] = 'SAS'
            self.AL_manifest.attrs['components'] = self.AL_components

            #self.AL_manifest  = self.AL_manifest.afl.comp.add_grid(pts_per_row=pts_per_row)

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
            
            






   
