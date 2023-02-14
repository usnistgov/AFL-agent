import pandas as pd
import numpy as np
import copy
import scipy.spatial
import random
import logging
from AFL.agent import PhaseMap
import matplotlib.pyplot as plt
from random import shuffle

from sklearn.metrics import pairwise_distances
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
        self.y_mean = None
        self.y_var = None
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

class Variance(Acquisition):
    def __init__(self):
        super().__init__()
        self.name = 'variance'
        
    def calculate_metric(self,GP):
        if self.phasemap is None:
            raise ValueError('No phase map set for acquisition! Call reset_phasemap!')
            
        self.y_mean,self.y_var = GP.predict(self.phasemap.afl.comp.get_grid(self.phasemap.attrs['components']).values)
        self.phasemap['acq_metric'] = ('grid',self.y_var.sum(1))
        self.phasemap.attrs['acq_metric'] = self.name

        return self.phasemap
    
class Random(Acquisition):
    def __init__(self):
        super().__init__()
        self.name = 'random'
        
    def calculate_metric(self,GP):
        if self.phasemap is None:
            raise ValueError('No phase map set for acquisition! Call reset_phasemap!')
            
        self.y_mean,self.y_var = GP.predict(self.phasemap.afl.comp.get_grid(self.phasemap.attrs['components_grid']).values)
            
        indices = np.arange(self.phasemap['grid'].shape[0])
        random.shuffle(indices)
        self.phasemap['acq_metric'] = ('grid',indices)
        self.phasemap.attrs['acq_metric'] = self.name
        return self.phasemap
    
class IterationCombined(Acquisition):
    def __init__(self,function1,function2,function2_frequency=5):
        super().__init__()
        self.function1 = function1
        self.function2 = function2
        # self.name = 'IterationCombined'  
        # self.name += '-'+function1.name
        # self.name += '-'+function2.name
        self.name = function1.name
        self.name += '-'+function2.name
        self.name += '@'+str(function2_frequency)
        self.iteration = 1
        self.function2_frequency=function2_frequency
        
    def calculate_metric(self,GP):
        if self.function1.phasemap is None:
            raise ValueError('No phase map set for acquisition! Call reset_phasemap!')

        self.y_mean,self.y_var = GP.predict(self.phasemap.afl.comp.get_grid(self.phasemap.attrs['components_grid']).values)
        
        if ((self.iteration%self.function2_frequency)==0):
            print(f'Using acquisition function {self.function2.name} of iteration {self.iteration}')
            self.phasemap = self.function2.calculate_metric(GP)
            self.phasemap.attrs['acq_current_metric'] = self.function1.name
        else:
            print(f'Using acquisition function {self.function1.name} of iteration {self.iteration}')
            self.phasemap = self.function1.calculate_metric(GP)
            self.phasemap.attrs['acq_current_metric'] = self.function2.name

        self.phasemap.attrs['acq_metric'] = self.name
        self.phasemap.attrs['acq_metric1'] = self.function1.name
        self.phasemap.attrs['acq_metric2'] = self.function2.name
        self.phasemap.attrs['acq_iteration'] = self.iteration
        self.iteration+=1
            
        return self.phasemap
    
class LowestDensity(Acquisition):
    def __init__(self,bandwidth=0.075):
        super().__init__()
        self.name = 'LowestDensity'
        self.bandwidth=bandwidth
        
    def calculate_metric(self,GP): 
        if self.phasemap is None:
            raise ValueError('No phase map set for acquisition! Call reset_phasemap!')
            
        #this must be calculated regardless of whether it is used
        self.y_mean,self.y_var = GP.predict(self.phasemap.afl.comp.get_grid(self.phasemap.attrs['components_grid']).values)
            
        xy = self.phasemap.afl.comp.to_xy()
        xy_grid = self.phasemap.afl.comp.to_xy(self.phasemap.attrs['components_grid'])
        kde = KernelDensity(bandwidth=self.bandwidth)
        kde.fit(xy)
        
        acq_metric = -np.exp(kde.score_samples(xy_grid))
        self.phasemap['acq_metric'] = ('grid',acq_metric)
        return self.phasemap
