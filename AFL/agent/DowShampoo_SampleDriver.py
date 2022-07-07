from AFL.automation.APIServer.Client import Client
from AFL.automation.prepare.OT2Client import OT2Client
from AFL.automation.shared.utilities import listify
from AFL.automation.APIServer.Driver import Driver
from AFL.automation.shared.Serialize import serialize,deserialize
from AFL.agent.AgentClient import AgentClient
from AFL.automation.shared.units import units

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
import warnings

import AFL.automation.prepare
from AFL.automation.shared.units import units
from AFL.automation import prepare
import shutil
import h5py


class SAS_AL_SampleDriver(Driver):
    defaults={}
    defaults['snapshot_directory'] = '/home/afl642'
    defaults['data_path'] = '/nfs/aux/chess/reduced_data/cycles/2022-1/id3b/beaucage-2324-D/analysis/'
    defaults['csv_data_path'] = '/nfs/aux/chess/reduced_data/cycles/2022-1/id3b/beaucage-2324-D/analysis/'
    defaults['data_manifest_file'] = '/nfs/aux/chess/reduced_data/cycles/2022-1/id3b/beaucage-2324-D/analysis/data_manifest.csv'
    defaults['data_tag'] = 'default'
    def __init__(self,
            load_url,
            prep_url,
            sas_url,
            agent_url,
            camera_urls = None,
            snapshot_directory =None,
            overrides=None, 
            dummy_mode=False,
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

        self.dummy_mode= dummy_mode
        if not self.dummy_mode:
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
        else:
            self.sas_url = sas_url
            self.prep_url = prep_url
        
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

        self.stocks = None
        self.stock_loc = None
        
    def status(self):
        status = []
        status.append(f'Snapshots: {self.config["snapshot_directory"]}')
        status.append(f'Cameras: {self.camera_urls}')
        status.append(f'SAS: {self.sas_url}')
        status.append(f'Sample Wait Time: {self.wait_time}')
        status.append(f'AL_components: {self.AL_components}')
        status.append(self.status_str)
        status.append(self.AL_status_str)
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

    def read_data(self,fname,iter_limit=300):
        self.app.logger.warning(f'Attempting to read {fname}...')
        i = 0
        while i<iter_limit:
            try:
                data = np.loadtxt(fname)
            except FileNotFoundError:
                time.sleep(1)
                i+=1
            else:
                break

        self.app.logger.warning(f'Read {fname}!')
        if i>=(iter_limit-1):
            raise ValueError('Timed out on file reading...')

        with open(fname,'r') as f:
            while True:
                line = f.readline()
                if 'Rayonix File Comments Template' in line:
                    headers = line.split('Rayonix File Comments Template')[1].replace('#','').strip().replace('*','').split(' ')
                    metadata = f.readline().split('Rayonix File Comments')[1].replace('#','').strip().replace('*','').split(' ')
                    break
        q = data[:,1]
        I = data[:,2]
        ds = xr.Dataset()
        ds['I'] = xr.DataArray(I,dims=['q'],coords={'q':q})
        ds['filename'] = fname.parts[-1]
        ds['detector'] = fname.parts[-1].split('_')[-3]
        ds['scan_num'] = int(fname.parts[-1].split('_')[-2])
        for k,v in zip(headers,metadata):
            try:
                ds[k] = float(v)
            except:
                ds[k] = v
        ds['T']  = ds['cntr3']/ds['cntr1']
        return ds

    def make_stocks(self):
        self.stocks = {}
        self.stock_loc = {}
        stock_H2O = prepare.Solution('stock_H2O',['H2O'])
        stock_H2O.volume = 20*units("ml")
        self.stocks['H2O'] = stock_H2O
        
        stock_SLES1 = prepare.Solution('stock_SLES1',['H2O','STEOLCS130'])
        stock_SLES1.volume = 20*units("ml")
        stock_SLES1.mass_fraction = {'STEOLCS130':0.25}
        stock_SLES1.density = 1.04*units('g/ml')
        self.stocks['SLES1'] = stock_SLES1
        
        stock_SLES3 = prepare.Solution('stock_SLES3',['H2O','STEOLCS330'])
        stock_SLES3.volume = 20*units("ml")
        stock_SLES3.mass_fraction = {'STEOLCS330':0.28}
        stock_SLES3.density = 1.03*units('g/ml')
        self.stocks['SLES3'] = stock_SLES3
        
        stock_CAPB = prepare.Solution('stock_CAPB',['H2O','AMPHOSOL'])
        stock_CAPB.volume = 20*units("ml")
        stock_CAPB.mass_fraction = {'AMPHOSOL':0.30}
        stock_CAPB.density = 1.04*units('g/ml')
        self.stocks['CAPB'] = stock_CAPB
        
        # stock4 = prepare.Solution('stock_BIOSOFT300',['H2O','BIOSOFT300'])
        # stock4.volume = 20*units("ml")
        # stock4.mass_fraction = {'BIOSOFT300':0.30}
        
        stock_DEX = prepare.Solution('stock_DEXTRAN',['H2O','DEXTRAN'])
        stock_DEX.volume = 20*units("ml")
        stock_DEX.mass_fraction = {'DEXTRAN':0.15}
        stock_DEX.density = 1.06*units('g/ml')
        self.stocks['DEX'] = stock_DEX

        self.stock_loc['SLES1'] = '4A1'
        self.stock_loc['SLES3'] = '4A2'
        self.stock_loc['CAPB'] = '4B2'
        self.stock_loc['DEX'] = '4B1'
        self.stock_loc['WATER'] = '1A1'

    def make_protocol(self,SLES1_SLES3_frac,CAPB_SLES_frac,CD1_pct,sample_mass,load_pipette_kw,load_pipette_fast_kw,prep_volume=500*units('ul'),load_volume=300*units('ul'),tot_surfactant=0.05):
        if self.stocks is None:
            self.make_stocks()

        # prep_volume = 500*units('ul')
        # load_volume = 300*units('ul')
        # sample_mass = (prep_volume*units('g/ml')).to('mg')
        # tot_surfactant = tot_surfactant*sample_mass
    
        SLES1_SLES3_frac = SLES1_SLES3_frac*units('')
        CAPB_SLES_frac = CAPB_SLES_frac*units('')
        CD1_pct = CD1_pct*units('')
    
        ##need to handle the case where SLES1_SLES3_frac==0
        if SLES1_SLES3_frac==0.:
            SLES1_mass = 0.0*units("mg")
            SLES3_mass = tot_surfactant/(1.0+CAPB_SLES_frac)
            CAPB_mass  = CAPB_SLES_frac*(SLES3_mass)
        else:
            SLES1_mass = tot_surfactant/(1.0 + 1.0/SLES1_SLES3_frac + CAPB_SLES_frac + CAPB_SLES_frac/SLES1_SLES3_frac)
            SLES3_mass = SLES1_mass/SLES1_SLES3_frac
            CAPB_mass  = CAPB_SLES_frac*(SLES1_mass+SLES3_mass)
        DEXTRAN_mass = CD1_pct*sample_mass
        WATER_mass = sample_mass - SLES1_mass - SLES3_mass - CAPB_mass - DEXTRAN_mass
    
        masses = {}
        masses['SLES1'] = SLES1_mass
        masses['SLES3'] = SLES3_mass
        masses['CAPB'] = CAPB_mass
        masses['DEX'] = DEXTRAN_mass
        masses['WATER'] = WATER_mass
    
    
        SLES1_stock_volume = (SLES1_mass/self.stocks['SLES1'].mass_fraction['STEOLCS130']/self.stocks['SLES1'].density).to('ul')
        SLES3_stock_volume = (SLES3_mass/self.stocks['SLES3'].mass_fraction['STEOLCS330']/self.stocks['SLES3'].density).to('ul')
        CAPB_stock_volume = (CAPB_mass/self.stocks['CAPB'].mass_fraction['AMPHOSOL']/self.stocks['CAPB'].density).to('ul')
        DEXTRAN_stock_volume = (DEXTRAN_mass/self.stocks['DEX'].mass_fraction['DEXTRAN']/self.stocks['DEX'].density).to('ul')
        WATER_stock_volume = (WATER_mass/(1.0*units('g/ml'))).to('ul')
    
        volumes = {}
        volumes['SLES1'] = SLES1_stock_volume
        volumes['SLES3'] = SLES3_stock_volume
        volumes['CAPB'] = CAPB_stock_volume
        volumes['DEXTRAN'] = DEXTRAN_stock_volume
        volumes['WATER'] = WATER_stock_volume
    
        ## Need to check volume limits
        skip = False
        names = ['DEX','WATER','SLES1','SLES3','CAPB']
        vols = [DEXTRAN_stock_volume,WATER_stock_volume,SLES1_stock_volume,SLES3_stock_volume,CAPB_stock_volume]
        for name,vol in zip(names,vols):
            if not pipette_possible(vol,name):
                skip=True
                break
        if skip:
            return False
    
        pa_list = []
    
        pa = prepare.PipetteAction(
            source=self.stock_loc['DEX'],
            dest='target',
            volume=DEXTRAN_stock_volume.magnitude,
            **load_pipette_kw)
        pa.kwargs['mix_after'] = None
        pa_list.append(pa)
    
        pa = prepare.PipetteAction(
            source=self.stock_loc['WATER'],
            dest='target',
            volume=WATER_stock_volume.magnitude,
            **load_pipette_fast_kw)
        pa_list.append(pa)
    
        pa = prepare.PipetteAction(
            source=self.stock_loc['SLES1'],
            dest='target',
            volume=SLES1_stock_volume.magnitude,
            **load_pipette_kw)
        pa_list.append(pa)
    
        pa = prepare.PipetteAction(
            source=self.stock_loc['SLES3'],
            dest='target',
            volume=SLES3_stock_volume.magnitude,
            **load_pipette_fast_kw)
        pa.kwargs['mix_after'] = None
        pa_list.append(pa)
    
        pa = prepare.PipetteAction(
            source=self.stock_loc['CAPB'],
            dest='target',
            volume=CAPB_stock_volume.magnitude,
            **load_pipette_fast_kw)
        pa_list.append(pa)
    
    
        protocol = []
        for pa in pa_list:
            if pa.kwargs['volume']>0.0:
                protocol.append(pa.emit_protocol())
    
        name = f'Dow_{SLES1_SLES3_frac.magnitude:3.2f}S1S3_{CAPB_SLES_frac.magnitude:3.2f}CS_{CD1_pct.magnitude*100:3.2f}CD'
        name = name.replace('.','p')
        return name,protocol,masses,volumes

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

        for task in sample['prep_protocol']:
            #if the well isn't in the map, just use the well
            task['source'] = target_map.get(task['source'],task['source'])
            task['dest'] = target_map.get(task['dest'],task['dest'])
            self.prep_uuid = self.prep_client.transfer(**task)
 
        if self.rinse_uuid is not None:
            self.update_status(f'Waiting for rinse...')
            self.load_client.wait(self.rinse_uuid)
            self.update_status(f'Rinse done!')

        self.update_status(f'Cell is clean, measuring empty cell scattering...')
        empty = {}
        empty['name'] = 'MT-'+sample['name']
        empty['exposure'] = sample['exposure']
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
    
    def active_learning_loop(self,**kwargs):
        self.AL_components = kwargs['AL_components']
        #sample_volume = kwargs['sample_volume']
        exposure = kwargs['exposure']
        #mix_order = kwargs['mix_order']
        #custom_stock_settings = kwargs['custom_stock_settings']
        data_manifest_path = pathlib.Path(self.config['data_manifest_file'])
        data_path = pathlib.Path(self.config['data_path'])
        csv_data_path = pathlib.Path(self.config['csv_data_path'])
        
        if self.dummy_mode:
            from AFL.automation.agent import PhaseMap
            df_measurements = pd.read_csv('/Users/tbm/projects/2001-NistoRoboto/2022-02-13-CodeDev/measurements.csv')
            
        self.stop_AL = False
        while not self.stop_AL:
            self.app.logger.info(f'Starting new AL loop')
            
            #get prediction of next step
            #next_sample = self.agent_client.get_next_sample_queued()
            next_sample = self.agent_client.get('next_sample')
            self.app.logger.info(f'Preparing to make next sample: {next_sample}')
            
            if self.dummy_mode:
                print(f'Pulling from {df_measurements.shape}')
                xy_all = PhaseMap.ternary2cart(df_measurements[self.AL_components].values)
                xy_next = PhaseMap.ternary2cart(next_sample[self.AL_components])
                dist = np.sqrt(np.sum(np.square(xy_all-xy_next),axis=1))
                argmin = np.argmin(dist)
                next_sample_full = df_measurements.iloc[argmin]
                next_sample = next_sample_full[self.AL_components].to_frame().T
                df_measurements = df_measurements.drop(argmin).reset_index(drop=True)
                print('\n\n\n')
                print(next_sample_full)
                print(df_measurements)
                print('\n\n\n')


            load_pipette_kw = {}
            load_pipette_kw['post_aspirate_delay'] = 10
            load_pipette_kw['post_dispense_delay'] = 10
            load_pipette_kw['aspirate_rate'] = 50
            load_pipette_kw['dispense_rate'] = 50
            load_pipette_kw['aspirate_equilibration_delay'] = 15.0
            load_pipette_kw['mix_after'] = (3,50)
            
            load_pipette_fast_kw = {}
            load_pipette_fast_kw['post_aspirate_delay'] = 5
            load_pipette_fast_kw['post_dispense_delay'] = 5
            load_pipette_fast_kw['aspirate_equilibration_delay'] = 2.
            load_pipette_fast_kw['aspirate_rate'] = 500
            load_pipette_fast_kw['dispense_rate'] = 50
            load_pipette_fast_kw['mix_after'] = (3,50)

            catch_protocol = prepare.PipetteAction(
                source='target',
                dest='10A1',
                volume = 300.,
                post_aspirate_delay = 10.,
                post_dispense_delay = 10.,
                aspirate_rate = 50.,
                dispense_rate = 50.,
                mix_aspirate_rate = 100.0,
                mix_dispense_rate = 100.0,
                mix_before=(5,300.),
            )

            prep_volume = 500*units('ul')
            load_volume = 300*units('ul')
            sample_mass = (prep_volume*units('g/ml')).to('mg')
            tot_surfactant = 0.05*sample_mass

            S1S3 = 1.0
            SLES3_frac = next_sample.sel(component='SLES3').values[0]
            CAPB_frac = next_sample.sel(component='CAPB').values[0]
            DEX_frac = next_sample.sel(component='DEX').values[0]
            self.app.logger.info(f'Preparing to make next sample with SLES3={SLES3_frac} CAPB={CAPB_frac} DEX={DEX_frac}')

            Cfr = CAPB_frac/(1-CAPB_frac)
            Dfr = DEX_frac/(1-DEX_frac)
            SLES3_mass = tot_surfactant/(1 + Cfr + S1S3 + Cfr*(Dfr+Cfr*Dfr)/(1-Cfr*Dfr))
            DEX_mass   = SLES3_mass*(Dfr + Dfr*Cfr)/(1-Dfr*Cfr)
            CAPB_mass  = tot_surfactant - SLES3_mass - SLES3_mass*S1S3
            SLES1_mass = SLES3_mass*S1S3
            
            CS = (CAPB_mass/(SLES3_mass+SLES1_mass)).magnitude
            CD = (DEX_mass/sample_mass).magnitude

            self.update_status(f'Converted next sample to be S1S3={S1S3:4.3f} CS={CS:4.3f} CD={CD:4.3f}')

            output = self.make_protocol(
                    S1S3,
                    CS,
                    CD,
                    sample_mass = sample_mass,
                    load_pipette_kw=load_pipette_kw,
                    load_pipette_fast_kw=load_pipette_fast_kw,
                    prep_volume=prep_volume,
                    load_volume=load_volume,
                    tot_surfactant=tot_surfactant
                    )

            if output==False:
                validated = False
            else:
                validated = True
                name,prep_protocol,masses,volumes = output
                DEX_mass   = SLES3_mass*(Dfr + Dfr*Cfr)/(1-Dfr*Cfr)
                CAPB_mass  = tot_surfactant - SLES3_mass - SLES3_mass*S1S3
                SLES1_mass = SLES3_mass*S1S3


            if validated:
                self.app.logger.info(f'Validation PASSED')
                self.AL_status_str = 'Last sample validation PASSED'
            else:
                self.app.logger.info(f'Validation FAILED')
                self.AL_status_str = 'Last sample validation FAILED'

            self.update_status(f'Making next sample to be S1S3={S1S3:4.3f} CS={CS:4.3f} CD={CD:4.3f}')
            
            if self.dummy_mode:
                sample_name = next_sample_full['fname'].replace('_r1d','')
                sample.target_check.mass_fraction
                sample.target_check.mass_fraction = next_sample_dict
                sample.target_check.volume = sample_volume*units('ul')
                print('SAMPLE_NAME:',sample_name)
                time.sleep(5)
            else:
                sample_uuid = str(uuid.uuid4())[-8:]
                sample_name = f'AL_{self.config["data_tag"]}_{sample_uuid}'
                self.process_sample(
                        dict(
                            name=sample_name,
                            prep_protocol = prep_protocol,
                            catch_protocol =[catch_protocol.emit_protocol()],
                            volume = catch_protocol.volume/1000.0,
                            exposure = exposure
                    )
                )
        

            data_files = list(data_path.glob(f'*{sample_uuid}*hs104*'))
            if len(data_files)==0:
                raise ValueError(f'No data files found...')
            elif len(data_files)>2:
                raise ValueError(f'Too many data files found:{data_files}')
            else:
                file_path = None
                for fname in data_files:
                    if 'MT-' not in str(fname):
                        file_path = fname
                        break
                if file_path is None:
                    raise ValueError(f"No non-MT data file found: {data_files}")

                
            # CHECK TRANMISSION OF LAST SAMPLE
            ds_data = self.read_data(file_path)
            transmission = ds_data['T']

            if transmission>0.85:
                self.update_status(f'Last sample missed! (Transmission={transmission})')
                self.app.logger.info('Dropping this sample from AL and hoping the next one hits...')
                continue
            else:
                self.update_status(f'Last Sample success! (Transmission={transmission})')

            # write csv to csv_data path
            pd_I = ds_data.I.to_pandas()
            pd_I.name = 'I'
            csv_fpath= csv_data_path/f'{sample_name}.csv'
            pd_I.to_csv(csv_fpath,header=False)

            
            # update manifest
            if data_manifest_path.exists():
                self.data_manifest = pd.read_csv(data_manifest_path)
            else:
                self.data_manifest = pd.DataFrame(columns=['fname','label',*self.AL_components])
            
            row = {}
            row['fname'] = csv_fpath.parts[-1]
            row['label'] = -1
            row['SLES3'] = SLES3_frac
            row['CAPB'] = CAPB_frac
            row['DEX'] = DEX_frac
            self.data_manifest = self.data_manifest.append(row,ignore_index=True)
            self.data_manifest.to_csv(data_manifest_path,index=False)
            
            # trigger AL
            self.agent_uuid = self.agent_client.enqueue(task_name='read_data',predict=True)
            
            # wait for AL
            self.app.logger.info(f'Waiting for agent...')
            self.agent_client.wait(self.agent_uuid)
            
            
      
   
def pipette_possible(volume,name):
    if (volume>1e-6*units('ul')) and (volume<5*units('ul')):
        print(f'--> Skipping {name} due to low volume:{volume}')
        return False
    else:
        return True





   
