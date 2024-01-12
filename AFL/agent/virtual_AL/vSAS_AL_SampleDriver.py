from AFL.automation.APIServer.Client import Client
from AFL.automation.prepare.OT2Client import OT2Client
from AFL.automation.shared.utilities import listify
from AFL.automation.APIServer.Driver import Driver
from AFL.agent.AgentClient import AgentClient
from AFL.automation.shared.units import units
from tiled.client import from_uri
from tiled.queries import Eq,Contains

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
    defaults['AL_manifest_file'] = '/nfs/aux/chess/reduced_data/cycles/2022-1/id3b/beaucage-2324-D/analysis/data_manifest.csv'
    defaults['data_tag'] = 'default'
    defaults['max_sample_transmission'] = 0.6
    defaults['mass_tolerance'] = 0.00
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
        
        #tiled
        self.tiled_url = tiled_url
        self.tiled_catalogue = from_uri(tiled_url,api_key='NistoRoboto642')
        

        if spec_url is not None:
            self.spec_url = spec_url
            self.spec_client = Client(spec_url.split(':')[0],port=spec_url.split(':')[1])
            self.spec_client.login('SampleServer_LSClient')
            self.spec_client.debug(False)
            self.spec_client.enqueue(task_name='setExposure',time=0.1)
        else:
            self.spec_client = None
    
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
        # self.wait_time = 30.0 #seconds
        self.wait_time = 0.01
        
        
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

        if self.spec_client is not None:
            self.spec_client.set_config(
                    filepath='/home/pi/2305_SINQ_data_reduced/',
                    filename=f'{sample["name"]}-beforeSAS-spec.h5'
                )
            self.spec_client.enqueue(task_name='collectContinuous',duration=5,interactive=False)

        ### the point in which the virtual SAS data driver will produce a spectrum and store it to tiled
        print('measuring')
        
        sas_uuid = self.sas_client.enqueue(task_name='expose',name=sample['name'],block=True,exposure=exposure)
        # self.sas_client.wait(sas_uuid)

        if self.spec_client is not None:
            self.spec_client.set_config(
                    filepath='/home/pi/2305_SINQ_data_reduced/',
                    filename=f'{sample["name"]}-afterSAS-spec.h5'
                )
            spec_uuid = self.spec_client.enqueue(task_name='collectContinuous',duration=5,interactive=False)
            # self.spec_client.wait(spec_uuid)

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

        ######
        # here is where we want to specify the data['sample_composition'] field. 
        ######
        
        
                
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
            # self.load_client.wait(self.rinse_uuid)
            self.update_status(f'Rinse done!')

        if self.spec_client is not None:
            self.spec_client.set_config(
                    filepath='/home/pi/2305_SINQ_data_reduced/',
                    filename=f'{sample["name"]}-MT-spec.h5'
                )
            self.spec_client.enqueue(task_name='collectContinuous',duration=5,interactive=False)

        # self.update_status(f'Cell is clean, measuring empty cell scattering...')
#         empty = {}
#         empty['name'] = 'MT-'+sample['name']
#         empty['exposure'] = sample['empty_exposure']
#         empty['wiggle'] = False
#         self.sas_uuid = self.measure(empty)
#         self.sas_client.wait(self.sas_uuid)
            
        if self.prep_uuid is not None: 
            # self.prep_client.wait(self.prep_uuid)
            self.take_snapshot(prefix = f'02-after-prep-{name}')

        
        self.update_status(f'Queueing sample {name} load into syringe loader')
        for task in sample['catch_protocol']:
            #if the well isn't in the map, just use the well
            task['source'] = target_map.get(task['source'],task['source'])
            task['dest'] = target_map.get(task['dest'],task['dest'])
            self.catch_uuid = self.prep_client.transfer(**task)
        
        if self.catch_uuid is not None:
            self.update_status(f'Waiting for sample prep/catch of {name} to finish: {self.catch_uuid}')
            # self.prep_client.wait(self.catch_uuid)
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
        # self.load_client.wait(self.load_uuid)
        self.take_snapshot(prefix = f'05-after-load-{name}')
        
        self.update_status(f'Sample is loaded, asking the instrument for exposure...')
        self.sas_uuid = self.measure(sample)

        self.update_status(f'Cleaning up sample {name}...')
        self.rinse_uuid = self.load_client.enqueue(task_name='rinseCell')

        self.update_status(f'Waiting for instrument to measure scattering of {name} with UUID {self.sas_uuid}...')
        # self.sas_client.wait(self.sas_uuid)
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
            # self.load_client.wait(self.load_uuid)
            last_name = self.last_sample['name']
            self.take_snapshot(prefix = f'05-after-load-{last_name}')

            self.update_status(f'Sample is loaded, asking the instrument for exposure...')
            self.sas_uuid = self.measure(self.last_sample)#self.measure blocks...
            self.take_snapshot(prefix = f'06-after-measure-{last_name}')
            
            self.update_status(f'Cleaning up sample {last_name}...')
            self.rinse_uuid = self.load_client.enqueue(task_name='rinseCell')
            # self.load_client.wait(self.rinse_uuid)

            self.update_status(f'Waiting for rinse...')
            # self.load_client.wait(self.rinse_uuid)
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
        # grid must have units for each component specified, hopefully this will progate to next_sample....
        self.components     = kwargs['components']
        self.AL_components  = kwargs['AL_components']
        self.AL_kwargs      = kwargs.get('AL_kwargs',{})
        self.fixed_concs    = kwargs.get('fixed_concs',{})
        sample_volume       = kwargs['sample_volume']
        sample_volume_units = kwargs['sample_volume_units']
        sample_volume       = sample_volume*units(sample_volume_units)
        exposure            = kwargs['exposure']
        empty_exposure      = kwargs['empty_exposure']
        predict             = kwargs.get('predict',True)
        start_new_manifest  = kwargs.get('start_new_manifest',False)
        ternary             = kwargs['ternary']#should be bool
        pre_run_list        = copy.deepcopy(kwargs.get('pre_run_list',[]))
        stop_after          = kwargs['stop_after']

        mix_order = kwargs.get('mix_order',None)
        custom_stock_settings = kwargs.get('custom_stock_settings',None)

        AL_manifest_path = pathlib.Path(self.config['AL_manifest_file'])
        data_path = pathlib.Path(self.config['csv_data_path'])
        
        self.stop_AL = False
        while not self.stop_AL:
            print(f'stop after iteration {stop_after}')
            print(f'current AL iteration {self.agent_client.get_driver_object("iteration")}')
            if self.agent_client.get_driver_object('iteration') >= stop_after:
                self.stop_AL = True
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
            if ternary:
                conc_spec = {}
                for name,value in self.fixed_concs.items():
                    conc_spec[name] = value['value']*units(value['units'])
                
                mass_dict = self.mfrac_to_mass(
                    mass_fractions=next_sample_dict,
                    specified_conc=conc_spec,
                    sample_volume=sample_volume,
                    output_units='mg')
            else:
                #assume concs for now...
                if len(next_sample_dict)<(len(self.components)-1):
                    raise ValueError('System under specified...')

                mass_dict = {}
                for name,comp in next_sample_dict.items():
                    mass_dict[name] = (comp['value']*units(comp['units'])*sample_volume).to('mg')
            
            self.target = AFL.automation.prepare.Solution('target',self.components)
            self.target.volume = sample_volume
            for k,v in mass_dict.items():
                print('mass dictionary')
                print(k,v)
                self.target[k].mass = v
                
            self.target.volume = sample_volume
            

            ######################
            ## MAKE NEXT SAMPLE ##
            ######################
            self.deck.reset_targets()
            self.deck.add_target(self.target,name='target')
            self.deck.make_sample_series(reset_sample_series=True)
            self.deck.validate_sample_series(tolerance=self.config['mass_tolerance'])
            
            self.deck.make_protocol(only_validated=False)
            if (mix_order is None) or (custom_stock_settings is None):
                warnings.warn(f'No mix_order or custom_stock_settings applied as mix_order={mix_order} and custom_stock_settings={custom_stock_settings}')
            else: 
                self.fix_protocol_order(mix_order,custom_stock_settings)
            self.sample,validated = self.deck.sample_series[0]
            
            ######################
            ## Validation Check ##
            ######################
            if validated:
                print(self.sample)
            
            
            ##################################
            ## BUILD DATASET FOR NEW SAMPLE ##
            ##################################
            self.new_data              = xr.Dataset()
            self.new_data['label']     = -1
            self.new_data['validated'] = validated
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
                    sample_composition[component]['values'] = self.new_data[component]
                    sample_composition[component]['units'] = 'fraction'
            else:
                for component in self.AL_components:
                    self.new_data[component] = self.sample.target_check.concentration[component].to("mg/ml").magnitude
                    self.new_data[component].attrs['units'] = 'mg/ml'
                    
                    sample_composition[component] = {}
                    sample_composition[component]['values'] = self.new_data[component].values.tolist()
                    sample_composition[component]['units'] = 'mg/ml'
                    
                
            for component in self.components:
                self.new_data['mfrac_'+component] = self.sample.target_check.mass_fraction[component].magnitude
                self.new_data['mass_'+component] = self.sample.target_check[component].mass.to('mg').magnitude
                self.new_data['mass_'+component].attrs['units'] = 'mg'
                    
            
            self.app.logger.info(self.deck.validation_report)
            
            if validated:
                self.app.logger.info(f'Validation PASSED')
                self.AL_status_str = 'Last sample validation PASSED'
            else:
                self.app.logger.info(f'Validation FAILED')
                self.AL_status_str = 'Last sample validation FAILED'
                
            #re-run the mass balance calculation here
            self.app.logger.info(f'Making next sample with mass fraction: {self.sample.target_check.mass_fraction}')
            
            self.catch_protocol.source = self.sample.target_loc
            
            sample_uuid = str(uuid.uuid4())
            sample_name = f'AL_{self.config["data_tag"]}_{sample_uuid[-8:]}'
            
            self.sas_client.set_config(
                filename=sample_name+'.h5'
            )

            print(sample_composition)
            
            sample_data = self.set_sample(sample_name=sample_name,sample_uuid=sample_uuid, sample_composition=sample_composition)
            self.prep_client.enqueue(task_name='set_sample',**sample_data)
            self.load_client.enqueue(task_name='set_sample',**sample_data)
            self.sas_client.enqueue(task_name='set_sample',**sample_data)
            self.agent_client.enqueue(task_name='set_sample',**sample_data)
            if self.spec_client is not None:
                self.spec_client.enqueue(task_name='set_sample',**sample_data)

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


            ################################
            ## VERIFY SAMPLE TRANSMISSION ##
            ################################
            
            #can bypass this for now but the data is not stored locally except in tiled....
            h5_path = data_path / (sample_name+'.h5')
            # with h5py.File(h5_path,'r') as h5:
            #     transmission = 0.1
                # transmission = h5['entry/sample/transmission'][()]
            transmission = 0.1
            if transmission>self.config['max_sample_transmission']:
                self.update_status(f'Last sample missed! (Transmission={transmission})')
                self.app.logger.info('Dropping this sample from AL and hoping the next one hits...')
                continue
            else:
                self.update_status(f'Last Sample success! (Transmission={transmission})')

            
            data_fname = sample_name+'.h5'
            # measurement = pd.read_csv(data_path/data_fname,sep=',',comment='#',header=None,names=['q','I'],usecols=[0,1]).set_index('q').squeeze().to_xarray()
            # measurement = measurement.dropna('q')
            
            query = self.tiled_catalogue.search(Eq('task_name','expose'))
            print(query)
            key, value = query.items()[-1]
            I = np.array(value)
            q = value.metadata['q']
            self.new_data['fname']     = data_fname
            self.new_data['SAS']       = xr.DataArray(data=I,dims=['q'],coords={'q':q}).dropna('q')
            
            self.new_data['transmission'] = transmission
            self.new_data['sample_uuid']      = sample_uuid
            
            ########################
            ## UPDATE AL MANIFEST ##
            ########################
            print()
            print()
            print()
            print()
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
                    self.agent_uuid = self.agent_client.enqueue(task_name='predict',datatype='nc',sample_uuid=sample_uuid)
                
                    # wait for AL
                    self.app.logger.info(f'Waiting for agent...')
                    # self.agent_client.wait(self.agent_uuid)
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
      
   






   
