from AFL.automation.APIServer.Client import Client
from AFL.automation.prepare.OT2Client import OT2Client
from AFL.automation.shared.utilities import listify
from AFL.automation.APIServer.Driver import Driver
from AFL.agent.AgentClient import AgentClient
from AFL.automation.shared.units import units

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


class SAS_AL_SampleDriver(Driver):
    defaults={}
    defaults['snapshot_directory'] = '/home/nistoroboto'
    defaults['csv_data_path'] = '/nfs/aux/chess/reduced_data/cycles/2022-1/id3b/beaucage-2324-D/analysis/'
    defaults['data_manifest_file'] = '/nfs/aux/chess/reduced_data/cycles/2022-1/id3b/beaucage-2324-D/analysis/data_manifest.csv'
    defaults['data_tag'] = 'default'
    defaults['max_sample_transmission'] = 0.6
    def __init__(self,
            load_url,
            prep_url,
            sas_url,
            agent_url,
            camera_urls = None,
            snapshot_directory =None,
            overrides=None, 
            ):

        Driver.__init__(self,name='SAS_AL_SampleDriver',defaults=self.gather_defaults(),overrides=overrides)

        if not (len(load_url.split(':'))==2):
            raise ArgumentError('Need to specify both ip and port on load_url')

        if not (len(prep_url.split(':'))==2):
            raise ArgumentError('Need to specify both ip and port on prep_url')

        if not (len(agent_url.split(':'))==2):
            raise ArgumentError('Need to specify both ip and port on agent_url')

        self.app = None
        self.name = 'SAS_AL_SampleDriver'

        #prepare samples
        self.prep_url = prep_url
        self.prep_client = OT2Client(prep_url.split(':')[0],port=prep_url.split(':')[1])
        self.prep_client.login('SampleServer_PrepClient')
        self.prep_client.debug(False)

        #load samples
        self.load_client = Client(load_url.split(':')[0],port=load_url.split(':')[1])
        self.load_client.login('SampleServer_LoadClient')
        self.load_client.debug(False)
# 
        #load samples
        self.sas_url = sas_url
        self.sas_client = Client(sas_url.split(':')[0],port=sas_url.split(':')[1])
        self.sas_client.login('SampleServer_SASClient')
        self.sas_client.debug(False)
        
        #agent
        self.agent_url = agent_url
        self.agent_client = AgentClient(agent_url.split(':')[0],port=agent_url.split(':')[1])
        self.agent_client.login('SampleServer_AgentClient')
        self.agent_client.debug(False)

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
        self.fixed_concs = None


        #XXX need to make deck inside this object because of 'different registries error in Pint
        self.reset_deck()
       
    def reset_deck(self):
        self.deck = AFL.automation.prepare.Deck()
        
    def add_container(self,name,slot):
        self.deck.add_container(name,slot)
        
    def add_catch(self,name,slot):
        self.deck.add_catch(name,slot)
        self.catch_loc=f"{slot}A1"
    
    def add_pipette(self,name,mount,tipracks):
        self.deck.add_pipette(name,mount,tipracks=tipracks)
        
    def send_deck_config(self,home=True):
        self.deck.init_remote_connection(
            self.prep_url.split(":")[0],
            home=home
        )
        self.deck.send_deck_config()
        
    def add_stock(self,stock_dict,loc):
        soln = AFL.automation.prepare.Solution.from_dict(stock_dict)
        self.deck.add_stock(soln,loc)
        
       
    def status(self):
        status = []
        status.append(f'Snapshots: {self.config["snapshot_directory"]}')
        status.append(f'Cameras: {self.camera_urls}')
        status.append(f'SAS: {self.sas_url}')
        status.append(f'Sample Wait Time: {self.wait_time}')
        status.append(f'AL_components: {self.AL_components}')
        status.append(f'{len(self.deck.stocks)} stocks loaded!')
        status.append(self.status_str)
        status.append(self.AL_status_str)
        status.append(f'Components: {self.components}')
        status.append(f'AL Components: {self.AL_components}')
        status.append(f'Fixed concs: {self.fixed_concs}')
        return status
    
    def update_status(self,value):
        self.status_str = value
        self.app.logger.info(value)

    def take_snapshot(self,prefix):
        now = datetime.datetime.now().strftime('%y%m%d-%H:%M:%S')
        for i,cam_url in enumerate(self.camera_urls):
            fname = self.config['snapshot_directory'] + '/' 
            fname += prefix
            fname += f'-{i}-'
            fname += now
            fname += '.jpg'

            try:
                r = requests.get(cam_url,stream=True)
                if r.status_code == 200:
                    with open(fname,'wb') as f:
                        r.raw.decode_content=True
                        shutil.copyfileobj(r.raw,f)
            except Exception as error:
                output_str  = f'take_snapshot failed with error: {error.__repr__()}\n\n'+traceback.format_exc()+'\n\n'
                self.app.logger.warning(output_str)

    def measure(self,sample):
        exposure = sample.get('exposure',None)

        sas_uuid = self.sas_client.enqueue(task_name='expose',name=sample['name'],block=True,exposure=exposure)
        self.sas_client.wait(sas_uuid)

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
            if i==(len(sample['prep_protocol'])-1):#last prepare
                task['drop_tip']=False
            self.prep_uuid = self.prep_client.transfer(**task)
 
        if self.rinse_uuid is not None:
            self.update_status(f'Waiting for rinse...')
            self.load_client.wait(self.rinse_uuid)
            self.update_status(f'Rinse done!')

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


    def process_sample_outoforder(self,sample):
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

        for task in sample['prep_protocol']:
            #if the well isn't in the map, just use the well
            task['source'] = target_map.get(task['source'],task['source'])
            task['dest'] = target_map.get(task['dest'],task['dest'])
            self.prep_uuid = self.prep_client.transfer(**task)
        
        if self.load_uuid is not None:
            self.load_client.wait(self.load_uuid)
            last_name = self.last_sample['name']
            self.take_snapshot(prefix = f'05-after-load-{last_name}')

            self.update_status(f'Sample is loaded, asking the instrument for exposure...')
            self.sas_uuid = self.measure(self.last_sample)#self.measure blocks...
            self.take_snapshot(prefix = f'06-after-measure-{last_name}')
            
            self.update_status(f'Cleaning up sample {last_name}...')
            self.rinse_uuid = self.load_client.enqueue(task_name='rinseCell')
            self.load_client.wait(self.rinse_uuid)

            self.update_status(f'Waiting for rinse...')
            self.load_client.wait(self.rinse_uuid)
            self.update_status(f'Rinse done!')
            
        self.update_status(f'Queueing sample {name} load into syringe loader')
        for task in sample['catch_protocol']:
            #if the well isn't in the map, just use the well
            task['source'] = target_map.get(task['source'],task['source'])
            task['dest'] = target_map.get(task['dest'],task['dest'])
            self.catch_uuid = self.prep_client.transfer(**task)
    
    @Driver.unqueued()
    def stop_active_learning(self):
        self.stop_AL = True
        
    def set_catch_protocol(self,**kwargs): 
        self.catch_protocol = AFL.automation.prepare.PipetteAction(**kwargs)
    
    
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
    
    def active_learning_loop(self,**kwargs):
        self.components = kwargs['components']
        self.AL_components = kwargs['AL_components']
        self.fixed_concs = kwargs['fixed_concs']
        sample_volume = kwargs['sample_volume']
        sample_volume_units = kwargs['sample_volume_units']
        sample_volume = sample_volume*units(sample_volume_units)
        exposure = kwargs['exposure']
        empty_exposure = kwargs['empty_exposure']
        predict = kwargs.get('predict',True)
        pre_run_list = copy.deepcopy(kwargs.get('pre_run_list',[]))

        mix_order = kwargs.get('mix_order',None)
        custom_stock_settings = kwargs.get('custom_stock_settings',None)

        data_manifest_path = pathlib.Path(self.config['data_manifest_file'])
        data_path = pathlib.Path(self.config['csv_data_path'])
        
        self.stop_AL = False
        while not self.stop_AL:
            self.app.logger.info(f'Starting new AL loop')
            
            #get prediction of next step
            if pre_run_list:
                self.next_sample = None
                next_sample_dict = pre_run_list.pop(0)
            else:
                #next_sample = self.agent_client.get_next_sample_queued()
                self.next_sample = self.agent_client.get_object('next_sample')
                next_sample_dict = self.next_sample.squeeze().reset_coords('grid',drop=True).to_pandas().to_dict()
            self.app.logger.info(f'Preparing to make next sample: {next_sample_dict}')

            conc_spec = {}
            for name,value in self.fixed_concs.items():
                conc_spec[name] = value['value']*units(value['units'])
            
            mass_dict = self.mfrac_to_mass(
                mass_fractions=next_sample_dict,
                specified_conc=conc_spec,
                sample_volume=sample_volume,
                output_units='mg')
            
            self.target = AFL.automation.prepare.Solution('target',self.components)
            self.target['D2O'].volume = sample_volume
            for k,v in mass_dict.items():
                self.target[k].mass = v
            
            self.deck.reset_targets()
            self.deck.add_target(self.target,name='target')
            self.deck.make_sample_series(reset_sample_series=True)
            self.deck.validate_sample_series(tolerance=0.15)
            self.deck.make_protocol(only_validated=False)
            if (mix_order is None) or (custom_stock_settings is None):
                warnings.warn('No mix_order or custom_stock_settings applied as mix_order={mix_order} and custom_stock_settings={custom_stock_settings}')
            else: 
                self.fix_protocol_order(mix_order,custom_stock_settings)
            self.sample,validated = self.deck.sample_series[0]
            self.app.logger.info(self.deck.validation_report)
            
            if validated:
                self.app.logger.info(f'Validation PASSED')
                self.AL_status_str = 'Last sample validation PASSED'
            else:
                self.app.logger.info(f'Validation FAILED')
                self.AL_status_str = 'Last sample validation FAILED'
            self.app.logger.info(f'Making next sample with mass fraction: {self.sample.target_check.mass_fraction}')
            
            self.catch_protocol.source = self.sample.target_loc
            
            sample_uuid = str(uuid.uuid4())[-8:]
            sample_name = f'AL_{self.config["data_tag"]}_{sample_uuid}'
            self.process_sample(
                    dict(
                        name=sample_name,
                        prep_protocol = self.sample.emit_protocol(),
                        catch_protocol =[self.catch_protocol.emit_protocol()],
                        volume = self.catch_protocol.volume/1000.0,
                        exposure = exposure,
                        empty_exposure = empty_exposure
                )
            )


            # warnings.warn('Transmission check not implemented. NOT USING TRANSMISSIONS TO CHECK FOR MISSES!',stacklevel=2)
            # # CHECK TRANMISSION OF LAST SAMPLE
            # # XXX Need to update based on how files will be read
            # file_path = data_path / (sample_name+'.txt')
            h5_path = data_path / (sample_name+'.h5')
            with h5py.File(h5_path,'r') as h5:
                transmission = h5['entry/sample/transmission'][()]
  
            if transmission>self.config['max_sample_transmission']:
                self.update_status(f'Last sample missed! (Transmission={transmission})')
                self.app.logger.info('Dropping this sample from AL and hoping the next one hits...')
                continue
            else:
                self.update_status(f'Last Sample success! (Transmission={transmission})')
            
            # update manifest
            if data_manifest_path.exists():
                self.data_manifest = pd.read_csv(data_manifest_path)
            else:
                AL_mfrac_comps = ['AL_mfrac_'+c for c in self.AL_components]
                mfrac_comps = ['mfrac_'+c for c in self.components]
                mass_comps = ['mass_'+c for c in self.components]
                self.data_manifest = pd.DataFrame(columns=['fname','label',*AL_mfrac_comps,*mfrac_comps,*mass_comps,'validated'])

            row = {}
            # row['fname'] = data_path/(sample_name+'_chosen_r1d.csv')
            row['fname'] = sample_name+'_chosen_r1d.csv'
            row['label'] = -1
            row['validated'] = validated
            total = 0
            for component in self.AL_components:
                mf = self.sample.target_check.mass_fraction[component].magnitude
                row['AL_mfrac_'+component] = mf
                total+=mf
            for component in self.AL_components:
                row['AL_mfrac_'+component] = row['AL_mfrac_'+component]/total
                
            for component in self.components:
                row['mfrac_'+component] = self.sample.target_check.mass_fraction[component].magnitude
                row['mass_'+component] = self.sample.target_check[component].mass.to('mg').magnitude
            
            self.data_manifest = self.data_manifest.append(row,ignore_index=True)
            self.data_manifest.to_csv(data_manifest_path,index=False)

            # trigger AL
            if pre_run_list: #still have manually added samples to be run
                pass
            elif predict:
                self.agent_uuid = self.agent_client.enqueue(task_name='predict')
            
                # wait for AL
                self.app.logger.info(f'Waiting for agent...')
                self.agent_client.wait(self.agent_uuid)
            else:#used for intialization
                return
            
            
    def fix_protocol_order(self,mix_order,custom_stock_settings):
        mix_order = [self.deck.get_stock(i) for i in mix_order]
        mix_order_map = {loc:new_index for new_index,(stock,loc) in enumerate(mix_order)}
        for sample,validated in self.deck.sample_series:
            # if not validated:
            #     continue
            old_protocol = sample.protocol
            ordered_indices = list(map(lambda x: mix_order_map.get(x.source),sample.protocol))
            argsort = np.argsort(ordered_indices)
            new_protocol = list(map(sample.protocol.__getitem__,argsort))
            time_patched_protocol = []
            for entry in new_protocol:
                if entry.source in custom_stock_settings:
                    for setting,value in custom_stock_settings[entry.source].items():
                        entry.__setattr__(setting,value)
                time_patched_protocol.append(entry)
            sample.protocol = time_patched_protocol
      
   






   
