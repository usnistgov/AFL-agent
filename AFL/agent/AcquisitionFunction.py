import pandas as pd
import numpy as np
import copy
import scipy.spatial
import random
import logging
from AFL.agent import PhaseMap
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances

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
        self.metric_rtol = 0.01 #pct tolerance
    
    def reset_phasemap(self,phasemap,components):
        self.phasemap = phasemap
        self.components = components
        self.grid_components = [k+"_grid" for k in components]
        
    def reset_mask(self,mask):
        self.mask = mask

    def add_exclusion(self,points):
        pass
    
    def plot(self,**kwargs):
        self.phasemap.afl.comp.plot_continuous(components=self.grid_components,labels='acq_metric')
        self.phasemap.afl.comp.plot_discrete(components=self.components,set_labels=False)

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



        if self.mask is None:
            mask = slice(None)
        else:
            mask = self.mask

        print(f"Running get_next_sample with sample_randomly={sample_randomly}")
        while True:
            #  print(f"Running {nth} iteration")
            if nth>=self.phasemap.grid.shape[0]:
                raise ValueError(f'No next sample found! Searched {nth} iterations!')

            if sample_randomly:
                metric = self.phasemap.where(mask,drop=True)
                metric = metric.afl.comp.get_grid(self.components)
                composition = metric.isel(grid=np.random.choice(metric.grid.shape[0],size=1))
            else:
                metric = self.phasemap.where(mask,drop=True)
                metric = metric.sortby('acq_metric')
                max_val= metric['acq_metric'].max()
                is_close = np.isclose(metric['acq_metric'],max_val,rtol=self.metric_rtol,atol=0)
                metric_mask = metric['acq_metric'].copy(data=is_close)
                metric = metric.where(metric_mask,drop=True)
                metric = metric.afl.comp.get_grid(self.components)
                composition = metric.isel(grid=np.random.choice(metric.grid.shape[0],size=1))

            if composition_check is None:
                break #all done

            dist = pairwise_distances(composition_check,composition).squeeze()[()]
            if (dist<self.composition_atol).any():
                nth+=1
            else:
                break

            if nth>1000:
                raise ValueError('Find next sample failed to converge!')

        self.next_sample = composition
        return self.next_sample

class Variance(Acquisition):
    def __init__(self):
        super().__init__()
        self.name = 'variance'
        
    def calculate_metric(self,GP):
        if self.phasemap is None:
            raise ValueError('No phase map set for acquisition! Call reset_phasemap!')
            
        self.y_mean,self.y_var = GP.predict(self.phasemap.afl.comp.get_grid(self.components).values)
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
            
        self.y_mean,self.y_var = GP.predict(self.phasemap.afl.comp.get_grid(self.components).values)
            
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
        
    def reset_phasemap(self,phasemap,components):
        self.function1.reset_phasemap(phasemap,components)
        self.function2.reset_phasemap(phasemap,components)
        self.phasemap = phasemap
        self.components = components
        self.grid_components = [k+"_grid" for k in components]
        
    def reset_mask(self,mask):
        self.function1.reset_mask(mask)
        self.function2.reset_mask(mask)
        self.mask = mask
        
    def calculate_metric(self,GP):
        if self.function1.phasemap is None:
            raise ValueError('No phase map set for acquisition! Call reset_phasemap!')

        self.y_mean,self.y_var = GP.predict(self.phasemap.afl.comp.get_grid(self.components).values)
        
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
