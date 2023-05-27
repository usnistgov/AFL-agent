import pandas as pd
import numpy as np
import copy
import scipy.spatial
import random
import logging
from AFL.agent import PhaseMap
import matplotlib.pyplot as plt
from random import shuffle

from sklearn.neighbors import KernelDensity

#move dense_pm definition outside of this class
#move to driver,make settable here and in driver
#pass make to driver

def gauss2d(x=0, y=0, cx=0, cy=0, sx=1, sy=1, a=-1):
    return a / (2. * np.pi * sx * sy) * np.exp(-((x - cx)**2. / (2. * sx**2.) + (y - cy)**2. / (2. * sy**2.)))

class Acquisition:
    def __init__(self):
        self.phasemap = None
        self.mask = None
        self.y_mean_GPR = None
        self.y_var_GPR = None
        self.y_mean_GPC = None
        self.y_var_GPC = None
        self.next_sample = None
        self.logger = logging.getLogger()
        self.composition_atol = 0.015#absolute distance tolerace
        self.metric_rtol = 0.03 #pct tolerance
    
    def reset_phasemap(self,phasemap):
        self.phasemap = phasemap
        
    def add_exclusion(self,points):
        pass
    
    def plot(self,masked=False,**kwargs):
        if masked:
            pm1 = self.phasemap[[var for var in self.phasemap if ('grid' in self.phasemap[var].dims)]].where(self.phasemap.mask,drop=True).copy()
            pm = self.phasemap.copy()
            pm = pm.drop_dims('grid').update(pm1)
        else:
            pm = self.phasemap.copy()
            
        pm.afl.comp.plot_continuous(components=self.phasemap.attrs['components_grid'],labels='acq_metric')
        pm.afl.comp.plot_discrete(components=self.phasemap.attrs['components'],set_labels=False)
        
        if self.next_sample is not None:
            plt.plot(*PhaseMap.to_xy(np.array([self.next_sample.squeeze().values])).T,marker='x',color='r')
        return plt.gca()
        
    def copy(self):
        return copy.deepcopy(self)

    def execute(self):
        raise NotImplementedError('Subclasses must implement execute!')

    def get_next_sample(self,nth=0,composition_check=None):

        if np.all(np.isnan(np.unique(self.phasemap['acq_metric']))):
            sample_randomly = True
        else:
            sample_randomly = False

        if 'mask' not in self.phasemap:
            mask = slice(None)
        else:
            mask = self.phasemap.mask
            
        print("Creating ordered metric lists...")
        if sample_randomly:
            close_metric = (
                self.phasemap.sel(grid=mask,drop=False)
                [composition.attrs['components_grid']] #just take composition values
                .rename({k:k.replace('_grid','') for k in composition.attrs['components_grid']})
                .to_array('component')
            )
        else:
            metric = self.phasemap.sel(grid=mask,drop=True)#.copy()
            metric = metric[['acq_metric']+metric.attrs['components_grid']]
            metric = metric.sortby('acq_metric')
            
            #find samples within rtol of maximum metric value
            is_close = np.isclose(
                metric['acq_metric'],
                metric['acq_metric'].max(),
                rtol=self.metric_rtol,
                atol=0
            )
             
            close_metric = (
                metric.sel(grid=is_close,drop=False)
                [metric.attrs['components_grid']] #just take composition values
                .rename({k:k.replace('_grid','') for k in metric.attrs['components_grid']})
                .to_array('component')
            )

        close_metric_idex_list = list(close_metric.grid.values) #list of integers
        shuffle(close_metric_idex_list)
        
        print(f"Running get_next_sample with sample_randomly={sample_randomly}")
        i = 0
        while True:
            print(f"Running iteration {i}")
            
            if close_metric_idex_list:
                print(f"Getting random point within {self.metric_rtol:0.2f} of maximum...")
                idex = close_metric_idex_list.pop()
            else:
                raise ValueError('No gridpoint found that satisfies constraints! Try increasing composition_rtol!')
            
            composition = close_metric.sel(grid=idex)
            
            if composition_check is None:
                break #all done
        
            print(f"Verifying that gridpoint isn't on top of previous measurement...")
            dist = (composition_check - composition).pipe(np.square).sum('component').pipe(np.sqrt)
            #if (dist<self.composition_atol).any():
            if (dist<self.composition_atol).any():
                i+=1
            else:
                break
        
            if nth>=self.phasemap.grid.shape[0]:
                raise ValueError(f'No next sample found! Searched {nth} iterations!')
            elif nth>1000:
                raise ValueError('Find next sample failed to converge!')

        print(f"Found that gridpoint {idex} satisfies the acquistion function and all constraints")
        self.next_sample = composition
        return self.next_sample

class pseudoUCB(Acquisition):
    """
    This acquisition function works on two GPs only. One has to be a classifier and the other has to be a regressor
    This needs to be generalizable and requires some thought...
    """
    def __init__(self,scaling =1.0, Thompson_sampling=False):
        super().__init__()
        self.name = 'pseudo UCB'
        self.scaling = scaling
        self.Thompson_sampling = Thompson_sampling

    def calculate_metric(self,GP,GPR):
        if self.phasemap is None:
            raise ValueError('No phase maps set for acquisition! Call reset_phasemap!')

        #needs to have some flexibility. for every GP or GPR in this calculation, 
        classifier_prediction = GP.predict(self.phasemap.attrs['components_grid'])
        self.y_mean_GPC = classifier_prediction['mean']
        self.y_var_GPC = classifier_prediction['var']
        

        regressor_prediction = GPR.predict(self.phasemap.attrs['components_grid'])
        self.y_mean_GPR = regressor_prediction['mean']
        self.y_var_GPR  = regressor_prediction['var']
        
        if self.Thompson_sampling:
            regressor_sample = GPR.model.predict_f_samples(GPR.transform_domain(components=self.phasemap.attrs['components_grid']),1)
            self.phasemap['acq_metric'] = ('grid', regressor_sample.numpy().squeeze() + self.scaling * self.y_var_GPC.sum(1))
            self.phasemap.attrs['acq_metric'] = self.name + '_TS'
        else:
            self.phasemap['acq_metric'] = ('grid',self.y_mean_GPR.squeeze() + self.scaling * self.y_var_GPC.sum(1))
            self.phasemap.attrs['acq_metric'] = self.name

        return self.phasemap
