import numpy as np
import copy
import scipy.spatial

import sklearn.mixture
import sklearn.cluster
from sklearn.metrics import pairwise

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance as dist
from collections import Counter

import warnings 
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import defaultdict

try:
    import tensorflow as tf
    import keras.models 
except ImportError:
    warnings.warn('Keras not installed. KerasClassifier will not work!')



class PhaseLabeler:
    def __init__(self,params=None):
        self.labels = None
        if params is None:
            self.params = {'n_phases':2}
        else:
            self.params = params
            
    def copy(self):
        return copy.deepcopy(self)
        
    def __getitem__(self,index):
        return self.labels[index]
    
    def __array__(self,dtype=None):
        return np.array(self.labels).astype(dtype)

    def remap_labels_by_count(self):
        label_map ={}
        for new_label,(old_label,_) in enumerate(sorted(Counter(self.labels).items(),key=lambda x: x[1],reverse=True)):
            label_map[old_label]=new_label
        self.labels = list(map(label_map.get,self.labels))
        
    def label(self):
        raise NotImplementedError('Sub-classes must implement label!')
        
    def silhouette(self,W):
        X = W.copy()
        silh_dict = defaultdict(list)
        max_n = min(X.shape[0],15)
        for n_phases in range(2,max_n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._init(n_phases=n_phases)
                self._label(X)
            self.remap_labels_by_count()
            
            if len(np.unique(self.labels))==1:
                silh_scores = np.zeros(len(X))
            else:
                silh_scores = silhouette_samples(
                    1.0-X,
                    self,
                    metric='precomputed'
                )
            silh_dict['all_scores'].append(silh_scores)
            silh_dict['avg_scores'].append(silh_scores.mean())
            silh_dict['n_phases'].append(n_phases)
            silh_dict['labels'].append(self.labels.copy())
            # silh_dict['labelers'].append(self.copy())
    
        silh_avg = np.array(silh_dict['avg_scores'])
        found = False
        for cutoff in np.arange(0.85,0.4,-0.05):
            idx = np.where(silh_avg>cutoff)[0]
            if idx.shape[0]>0:
                idx = idx[-1]
                found=True
                break
                
        if not found:
            self.n_phases = 1
            self.labels=np.zeros(X.shape[0])
        else:
            self.n_phases = silh_dict['n_phases'][idx]
            self.labels = silh_dict['labels'][idx]
        self.silh_dict = silh_dict
        
class GaussianMixtureModel(PhaseLabeler):
    def __init__(self,params=None):
        super().__init__(params)
        self.name = f'GaussianMixtureModel'
        
    def label(self,phasemap,**params):
        self._init(**params)
        self.silhouette(phasemap.W.values)
        
    def _init(self,**params):
        if params:
            self.params.update(params)
        self.clf = sklearn.mixture.GaussianMixture(self.params['n_phases'])
        
    def _label(self,metric):
        self.clf.fit(metric)
        self.labels = self.clf.predict(metric)
        
class SpectralClustering(PhaseLabeler):
    def __init__(self,params=None):
        super().__init__(params)
        self.name = f'SpectralClustering'
        
    def label(self,phasemap,**params):
        self._init(**params)
        self.silhouette(phasemap.W.values)
        
    def _init(self,**params):
        if params:
            self.params.update(params)
            
        self.clf = sklearn.cluster.SpectralClustering(
            self.params['n_phases'],
            affinity = 'precomputed',  
            assign_labels="discretize",  
            random_state=0,  
            n_init = 1000
        )
        
    def _label(self,metric):
        self.clf.fit(metric)
        self.labels = self.clf.labels_

        
    
class DBSCAN(PhaseLabeler):#!!!
    def __init__(self,params=None):
        super().__init__(params)
        self.name = f'DBSCAN'
        if 'eps' not in self.params:
            self.params['eps'] = 0.05
        
    def label(self,phasemap,**params):
        if params:
            self.params.update(params)
            
        self.clf = sklearn.cluster.DBSCAN(
            eps=self.params['eps'],
            metric='precomputed',
            min_samples=1
        )
        #need something that works like distance, not similarty
        self.clf.fit(1.0-phasemap.W.values)
        self.labels = self.clf.labels_
        self.n_phases = len(np.unique(self.labels))
        return self.labels

class KerasClassifier(PhaseLabeler):
    def __init__(self,model_path,model_q,params=None,data_variable='deriv0'):
        super().__init__(params)
        self.name = f'KerasClassifier'

        self.model_path = model_path
        self.model_q = model_q
        self.clf = None
        self.data_variable = data_variable
        
    def label(self,phasemap,transpose_var='sample',**params):
        if self.clf is None:
            with tf.device('/CPU:0'):
                self.clf = keras.models.load_model(self.model_path)
        X_data = phasemap[self.data_variable].transpose(transpose_var,...).interp(logq=np.log10(self.model_q)).values
        self.labels = self.clf.predict(X_data).argmax(axis=1)
        self.n_phases = len(np.unique(self.labels))
