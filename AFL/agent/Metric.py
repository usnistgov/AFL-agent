import numpy as np
import copy
import scipy.spatial

from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance as dist
from collections import Counter


def listify(obj):
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        obj = [obj]
    return obj

class Metric:
    def __init__(self,data_variable='data',params=None,name=None):
        self.W = None
        if name is None:
            self.name = 'BaseMetric'
        else:
            self.name = name
        self.data_variable = data_variable
        if params is None:
            self.params = {}
        else:
            self.params = params
    
    def _get_X(self,dataset,transpose_var='sample'):
        X = dataset[listify(self.data_variable)].to_array().squeeze()
        X = X.transpose(transpose_var,...)#ensure that transpose_var is first dimension
        return X
    
    def to_dict(self):
        out = {'name':self.name,'data_variable':self.data_variable}
        out.update(self.params)
        return out
            
    def __repr__(self):
        return f'<Metric:{self.name}>'
    
    def copy(self):
        return copy.deepcopy(self)
    
    def __getitem__(self,index):
        return self.W[index]
    
    def __array__(self,dtype=None):
        return self.W.astype(dtype)
        
    def __mul__(self,other):
        if (self.W is None) or (other.W is None):
            raise ValueError('Must call .calculate before combining embeddings')
        new = self.copy()
        new.W = self.W*other.W
        return new
    
    def normalize1(self):
        metric = self.copy()
        diag = np.diag(metric.W)
        metric.W = metric.W/np.sqrt(np.outer(diag,diag))
        return metric
    
    def normalize2(self):
        metric = self.copy()
        diag = np.diag(metric.W)
        metric.W = (metric.W - metric.W.min())/(metric.W.max()-metric.W.min())
        return metric
    
    def calculate(self,*args,**kwargs):
        raise NotImplementedError('Sub-classes must implement calculate!')
        
        
class Similarity(Metric):
    def __init__(self,data_variable='data',params=None,name=None):
        super().__init__(data_variable,params,name)
        if name is None:
            self.name = f'Similarity:{params["metric"]}'
        
    def calculate(self,dataset,**params):
        if params:
            self.params.update(params)
            
        X = self._get_X(dataset)
        self.W = pairwise.pairwise_kernels(
            X, 
            filter_params=True,  
            **self.params
        )
        return self
    
class Distance(Metric):
    def __init__(self,data_variable='data',params=None,name=None):
        super().__init__(data_variable,params,name)
        if name is None:
            self.name = f'Distance:{params["metric"]}'
        
    def calculate(self,dataset,**params):
        if params:
            self.params.update(params)
            
        X = dataset[listify(self.data_variable)].to_array().squeeze()
        self.W = pairwise.pairwise_distances(
            X, 
            **self.params
        )
        return self

class MultiMetric(Metric):
    def __init__(self,metrics,data_variable='data',combine_by='prod',combine_by_powers=None,combine_by_coeffs=None,**params):
        super().__init__(data_variable,params)
        self.metrics=metrics
        self.update_name()
        
        self.combine_by=combine_by
        self.combine_by_powers = combine_by_powers
        self.combine_by_coeffs = combine_by_coeffs
        if self.combine_by=='prod':
            self.combine=self.prod
        elif self.combine_by=='sum':
            self.combine=self.sum
        else:
            raise ValueError('Combine by function not recognized. Must be "sum" or "prod"')
            
    def to_dict(self):
        out= {}
        for i,metric in enumerate(self.metrics):
            out[f'metric_{i}'] = metric.to_dict()
        return out
           
    def update_name(self):
        self.name=''
        for metric in self.metrics:
            self.name += metric.name+'-'
        self.name = self.name[:-1]#remove last '-'
    
    def prod(self,data_list):
        if self.combine_by_powers is not None:
            assert len(self.combine_by_powers)==len(data_list)
            powers = copy.deepcopy(self.combine_by_powers)
        else:
            powers = [1]*len(data_list)
        
        data_list = copy.deepcopy(data_list)
        
        value = np.power(data_list.pop(0),powers.pop(0))
        for data,power in zip(data_list,powers):
            value*=np.power(data,power)
        return value
    
    def sum(self,data_list):
        if self.combine_by_powers is not None:
            assert len(self.combine_by_powers)==len(data_list)
            coeffs = copy.deepcopy(self.combine_by_coeffs)
        else:
            coeffs = [1]*len(data_list)
        
        data_list = copy.deepcopy(data_list)
        
        value = data_list.pop(0)*coeffs.pop(0)
        for data,coeff in zip(data_list,coeffs):
            value += data*coeff
        return value
        
    def calculate(self,dataset):
        W_list = []
        for metric in self.metrics:
            W_list.append(metric.calculate(dataset))
        self.W_list = W_list
        self.W = self.combine(W_list)
        return self
    
class Delaunay(Similarity):
    def __init__(self,data_variable='data',params=None):
        super().__init__(data_variable,params)
        self.name = 'Delaunay'
        
    def calculate(self,X):
        """
        Computes the Delaunay triangulation of the given points
        :param x: array of shape (num_nodes, 2)
        :return: the computed adjacency matrix
        """
        tri = scipy.spatial.Delaunay(X)
        edges_explicit = np.concatenate((tri.vertices[:, :2],
                                         tri.vertices[:, 1:],
                                         tri.vertices[:, ::2]), axis=0)
        adj = np.zeros((x.shape[0], x.shape[0]))
        adj[edges_explicit[:, 0], edges_explicit[:, 1]] = 1.
        self.W = np.clip(adj + adj.T, 0, 1) 
        return self
    
    
