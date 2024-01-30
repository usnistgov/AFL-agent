import datetime
import pathlib
import uuid
import warnings

import gpflow
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from AFL.agent import GaussianProcess
from AFL.agent import Metric
from AFL.automation.APIServer.Driver import Driver
from AFL.automation.shared.utilities import mpl_plot_to_bytes


class DoubleAgentDriver(Driver):
    defaults={}
    defaults['compute_device'] = '/device:CPU:0'
    defaults['data_path'] = '~/'
    defaults['AL_manifest_file'] = 'manifest.nc'
    defaults['save_path'] = '/home/AFL/'
    defaults['data_tag'] = 'default'
    defaults['qlo'] = 0.001
    defaults['qhi'] = 1
    defaults['subtract_background'] = False

    def __init__(self,overrides=None):
        Driver.__init__(self,name='SAS_AgentDriver',defaults=self.gather_defaults(),overrides=overrides)

        self.watchdog = None 
        self.AL_manifest = None
        self._app = None
        self.name = 'DoubleAgentDriver'

        self.status_str = 'Fresh Server!'

        self.n_phases = None
        self.acquisition = None
        self.labeler = None
        self.next_sample = None
        self._mask = None

        self.dataset = None
        self.metrics =  None
        self.pipelines = None

        self.iteration = 0
        self.acq_count = 0

    @property
    def app(self):
        return self._app
    
    @app.setter
    def app(self,value):
        self._app = value

    def status(self):
        status = [self.status_string]

        if self.n_phases is not None:
            status.append(f'Found {self.n_phases} phases')

        status.append(f'Using {self.config["compute_device"]}')
        status.append(f'Data Manifest:{self.config["AL_manifest_file"]}')
        status.append(f'Iteration {self.iteration}')
        status.append(f'Acquisition Count {self.acq_count}')
        return status
    
    def update_status(self,value):
        self.status_str = value
        self.app.logger.info(value)
    
    def predict(self):
        self.read_dataset()
        self.preprocess_dataset()
        self.label()
        self.extrapolate()
        self.get_next_sample()
        self.save_results()

    def read_dataset(self):
        """Read and process a dataset from a netcdf file"""
        self.update_status(f'Reading the latest data in {self.config["AL_manifest_file"]}')
        self.dataset = xr.load_dataset(self.config['AL_manifest_file'])

    def preprocess_dataset(self):
        """Validate dataset and apply preprocessing pipelines"""

        assert 'dataset' in self.dataset.dims,'Must have "dataset" dim in dataset"
        assert 'components' in self.dataset,'components must be specified in dataset'
        assert 'AL_data' in self.dataset,'AL_data must be specified in dataset'
        assert 'AL_goals' in self.dataset,'AL_goals must be specified in dataset'
        assert 'AL_comps' in self.dataset,'AL_comps must be specified in dataset'

        for name, dataset in self.dataset.groupby('dataset'):
            AL_data = dataset['AL_data'].values[()]
            assert AL_data in self.dataset,f'AL_data {AL_data} not found in dataset'

            AL_comps = dataset['AL_comps'].values[()]
            assert AL_comps in self.dataset,f'AL_comps {AL_comps} not found in dataset'

            AL_sample_dims = dataset['AL_sample_dims'].values[()]
            assert AL_sample_dims in self.dataset, f'AL_sample_dims {AL_sample_dims} not found in dataset'


        for pipeline in self.pipelines:
            self.dataset = pipeline.calculate(self.dataset)

    def label(self):
        # handle combined W and clustering or separate

        # first calculate all metrics and dump into the dataset
        for i,metric in enumerate(listify(self.metrics)):
            metric.calculate(self.dataset)

            W_name = 'W_{metric.name}'
            self.dataset[W_name] = ((f'{W_name}_i',f'{W_name}_j'),metric.W)
            self.dataset[W_name].attrs['dictstr'] = str(metric.to_dict())

        for i,(name,dataset) in enumerate(self.dataset.groupby('dataset')):
            if not (dataset['AL_goals'].values[()]=='classification'):
                continue

            #XXX need to specify which dataset to calculate
            self.labeler.label(self.dataset)

            self.update_status(f'Found {self.labeler.n_phases} phases')

            self.dataset['{name}_labels'] = ('sample',self.labeler.labels)

        if self.data is not None:
            self.data['n_phases'] = self.n_phases
            self.data.add_array('labels',np.array(self.labeler.labels))
        
    def extrapolate(self):
        # Predict phase behavior at each point in the phase diagram
        if self.n_phases==1:
            self.update_status(f'Using dummy GP for one phase')
            self.GP = GaussianProcess.DummyGP(self.dataset)
        else:
            self.update_status(f'Starting gaussian process calculation on {self.config["compute_device"]}')
            with tf.device(self.config['compute_device']):
                self.kernel = gpflow.kernels.Matern32(variance=0.5,lengthscales=1.0) 
                self.GP = GaussianProcess.GP(
                    dataset = self.dataset,
                    kernel=self.kernel
                )
                self.GP.optimize(2000,progress_bar=True)
                
            
        if (self.data is not None) and not (self.n_phases==1):
            from gpflow.utilities.traversal import leaf_components
            for name,value in leaf_components(self.GP.model).items():
                value_numpy = value.numpy()
                if value_numpy.dtype=='object':
                    self.data[f'GP-{name}'] = value_numpy
                else:
                    self.data.add_array(f'GP-{name}',value_numpy)
                
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
        
        from xarray.core.merge import MergeError
        
        if 'component' in self.dataset.dims:
            warnings.warn('Dropping component dim (and all associated vars) from phasemap...')
            self.dataset = self.dataset.drop_dims('component')
            
        try:
            self.dataset['next_sample'] = self.acquisition.next_sample.squeeze()
        except MergeError:
            self.dataset['next_sample'] = self.acquisition.next_sample.squeeze().reset_coords(drop=True)
            
        if 'mask' in self.dataset:
            self.dataset['mask'] = self.dataset['mask'].astype(int)
        

        self.dataset.attrs['uuid'] = uuid_str
        self.dataset.attrs['date'] = date
        self.dataset.attrs['time'] = time
        self.dataset.attrs['data_tag'] = self.config["data_tag"]
        self.dataset.attrs['acq_count'] = self.acq_count
        self.dataset.attrs['iteration'] = self.iteration
        self.dataset.to_netcdf(save_path/f'phasemap_{self.config["data_tag"]}_{uuid_str}.nc')
        
        array_data = ['gp_y_var', 'gp_y_mean', 'acq_metric', '']
        if self.data is not None:
            for key,value in self.dataset.items():
                if value.dtype=='object':
                    self.data[key] = {'dims':value.dims,'values':value.values}
                else:
                    self.data.add_array(key,value.values)
                    self.data['array_dims'] = value.dims
        
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


    @Driver.unqueued(render_hint='precomposed_png')
    def plot_scatt(self,**kwargs):
        if self.dataset is not None:
            if 'labels_ordinal' not in self.dataset:
                self.dataset['labels_ordinal'] = ('system',np.zeros(self.dataset.sizes['sample']))
                labels = [0]
            else:
                labels = np.unique(self.dataset.labels_ordinal.values)

        print('render_hint=',kwargs['render_hint'])
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
                    spm.deriv0.afl.scatt.plot_linlin(x='logq',legend=False);
                
                    plt.sca(axes[i,1])
                    spm.afl.comp.plot_discrete(components=self.dataset.attrs['components']);
    
            img  = mpl_plot_to_bytes(fig,format=kwargs['render_hint'].split('_')[1])
            return img
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
                    out_dict[f'phase_{i}']['deriv0'] = list(spm.deriv0.values)
                    out_dict[f'phase_{i}']['compositions'] = {}
                    for component in self.dataset.attrs['components']:
                        out_dict[f'phase_{i}']['compositions'][component] = list(spm[component].values)
            return out_dict
        else:
            raise ValueError(f'Cannot handle render_hint={kwargs["render_hint"]}')

    @Driver.unqueued(render_hint='precomposed_png')
    def plot_acq(self,**kwargs):
        matplotlib.use('Agg') #very important
        fig,ax = plt.subplots()
        if self.dataset is None:
            plt.text(1,5,'No phasemap loaded. Run .read_data()')
            plt.gca().set(xlim=(0,10),ylim=(0,10))
        else:
            self.acquisition.plot()
        if 'precomposed' in kwargs['render_hint']:
            img  = mpl_plot_to_bytes(fig,format=kwargs['render_hint'].split('_')[1])
            return img
        else:
            raise ValueError(f'Cannot handle render_hint={kwargs["render_hint"]}')

    @Driver.unqueued(render_hint='precomposed_png')
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



    


   
