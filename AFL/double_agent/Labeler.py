"""
Phase labeling tools for clustering and classification of materials data.

This module provides classes for automatically identifying and labeling phases in
materials science data. It includes various clustering algorithms and methods for
determining optimal number of phases using silhouette analysis.

Key features:
- Multiple clustering algorithms (Spectral, GMM, Affinity Propagation)
- Automatic determination of number of phases via Silhouette analysis
- Support for precomputed similarity/distance matrices
- Integration with scikit-learn clustering algorithms
"""

import copy
import warnings
from collections import Counter
from collections import defaultdict

import numpy as np
import xarray as xr
import sklearn.cluster
import sklearn.mixture
from sklearn.metrics import silhouette_samples

from AFL.double_agent.PipelineOp import PipelineOp


class Labeler(PipelineOp):
    """Base class for phase labeling operations.
    
    This abstract base class provides common functionality for labeling phases
    in materials data. It supports various clustering approaches and includes
    methods for label manipulation and silhouette analysis.

    Parameters
    ----------
    input_variable : str
        The name of the variable containing the data to be labeled

    output_variable : str
        The name of the variable where labels will be stored

    dim : str, default='sample'
        The dimension name for samples in the dataset

    use_silhouette : bool, default=False
        Whether to use silhouette analysis to determine optimal number of phases

    params : dict, optional
        Additional parameters for the labeling algorithm

    name : str, default='PhaseLabeler'
        The name to use when added to a Pipeline
    """

    def __init__(self, input_variable, output_variable, dim='sample', use_silhouette=False, params=None, name='PhaseLabeler'):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.labels = None
        self.dim = dim

        if params is None:
            self.params = {}
        else:
            self.params = params

        self._banned_from_attrs.extend(['labels','clf','silh_dict','params'])
        self.use_silhouette = use_silhouette

    def __getitem__(self, index):
        """Get labels at specified index."""
        return self.labels[index]

    def __array__(self, dtype=None):
        """Convert labels to numpy array."""
        return np.array(self.labels).astype(dtype)

    def remap_labels_by_count(self):
        """Remap phase labels to be ordered by frequency.
        
        Reorders labels so that the most common phase is labeled 0,
        second most common is 1, etc.
        """
        label_map = {}
        for new_label, (old_label, _) in enumerate(
                sorted(Counter(self.labels).items(), key=lambda x: x[1], reverse=True)):
            label_map[old_label] = new_label
        self.labels = list(map(label_map.get, self.labels))

    def silhouette(self, W):
        """Perform silhouette analysis to determine optimal number of phases.
        
        Uses silhouette scores to automatically determine the best number of
        phases by trying different numbers of clusters and analyzing the
        clustering quality.

        Parameters
        ----------
        W : array-like
            The similarity/distance matrix to use for clustering

        Notes
        -----
        This method modifies the instance's n_phases and labels attributes
        based on the silhouette analysis results.
        """
        X = W.copy()
        silh_dict = defaultdict(list)
        max_n = min(X.shape[0], 15)
        for n_phases in range(2, max_n):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._construct_labeler(n_phases=n_phases)
                self._label(X)
            self.remap_labels_by_count()

            if len(np.unique(self.labels)) == 1:
                silh_scores = np.zeros(len(X))
            else:
                silh_scores = silhouette_samples(
                    1.0 - X,
                    self,
                    metric='precomputed'
                )
            silh_dict['all_scores'].append(silh_scores)
            silh_dict['avg_scores'].append(silh_scores.mean())
            silh_dict['n_phases'].append(n_phases)
            silh_dict['labels'].append(self.labels.copy())

        silh_avg = np.array(silh_dict['avg_scores'])
        found = False
        for cutoff in np.arange(0.85, 0.05, -0.05):
            idx = np.where(silh_avg > cutoff)[0]
            if idx.shape[0] > 0:
                idx = idx[-1]
                found = True
                break

        if not found:
            self.n_phases = 1
            self.labels = np.zeros(X.shape[0])
        else:
            self.n_phases = silh_dict['n_phases'][idx]
            self.labels = silh_dict['labels'][idx]
        self.silh_dict = silh_dict


class SpectralClustering(Labeler):
    """Spectral clustering for phase identification.
    
    Uses spectral clustering to identify phases based on a similarity matrix.
    Particularly effective for non-spherical clusters and when working with
    similarity rather than distance metrics.

    Parameters
    ----------
    input_variable : str
        The name of the variable containing the similarity matrix

    output_variable : str
        The name of the variable where labels will be stored

    dim : str
        The dimension name for samples in the dataset

    params : dict, optional
        Additional parameters for spectral clustering

    name : str, default='SpectralClustering'
        The name to use when added to a Pipeline

    use_silhouette : bool, default=False
        Whether to use silhouette analysis to determine optimal number of phases
    """

    def __init__(self, input_variable, output_variable, dim, params=None, name='SpectralClustering', use_silhouette=False):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable, dim=dim,
                         params=params, use_silhouette=use_silhouette)

    def _construct_labeler(self, **params):
        """Construct the spectral clustering model.
        
        Parameters
        ----------
        **params : dict
            Additional parameters to update the clustering configuration
        """
        self.params.update(params)

        self.clf = sklearn.cluster.SpectralClustering(
            self.params['n_phases'],
            affinity='precomputed',
            assign_labels="discretize",
            random_state=0,
            n_init=1000
        )

    def calculate(self, dataset):
        """Apply spectral clustering to the dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing the similarity matrix

        Returns
        -------
        Self
            The SpectralClustering instance with computed labels
        """
        data1 = self._get_variable(dataset)
        self._construct_labeler()
        if self.use_silhouette:
            self.silhouette(data1.values)
        else:
            self.labels = self.clf.fit_predict(data1.values)
        self.output[self.output_variable] = xr.DataArray(self.labels, dims=[self.dim])
        self.output[self.output_variable].attrs.update(self.params)
        return self

    def _label(self, metric):
        """Perform the actual clustering.
        
        Parameters
        ----------
        metric : array-like
            The similarity matrix to cluster
        """
        self.clf.fit(metric)
        self.labels = self.clf.labels_


class GaussianMixtureModel(Labeler):
    """Gaussian Mixture Model for phase identification.
    
    Uses a Gaussian Mixture Model to identify phases based on their distribution
    in feature space. Particularly useful when phases are expected to have
    Gaussian distributions.

    Parameters
    ----------
    input_variable : str
        The name of the variable containing the feature data

    output_variable : str
        The name of the variable where labels will be stored

    dim : str
        The dimension name for samples in the dataset

    params : dict, optional
        Additional parameters for the GMM

    name : str, default='GaussianMixtureModel'
        The name to use when added to a Pipeline
    """

    def __init__(self, input_variable, output_variable, dim, params=None, name='GaussianMixtureModel'):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable, dim=dim,
                         params=params)

    def _construct_labeler(self, **params):
        """Construct the GMM model.
        
        Parameters
        ----------
        **params : dict
            Additional parameters to update the GMM configuration
        """
        self.params.update(params)
        self.clf = sklearn.mixture.GaussianMixture(self.params['n_phases'])

    def calculate(self, dataset):
        """Apply GMM clustering to the dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing the feature data

        Returns
        -------
        Self
            The GaussianMixtureModel instance with computed labels
        """
        data1 = self._get_variable(dataset)
        self._construct_labeler()
        self.silhouette(data1.values)
        self.output[self.output_variable] = xr.DataArray(self.labels, dims=[self.dim])
        return self

    def _label(self, metric):
        """Perform the actual clustering.
        
        Parameters
        ----------
        metric : array-like
            The feature data to cluster
        """
        self.clf.fit(metric)
        self.labels = self.clf.predict(metric)


class AffinityPropagation(Labeler):
    """Affinity Propagation for phase identification.
    
    Uses Affinity Propagation clustering to identify phases. This method
    automatically determines the number of phases based on the data structure
    and similarity matrix.

    Parameters
    ----------
    input_variable : str
        The name of the variable containing the similarity matrix

    output_variable : str
        The name of the variable where labels will be stored

    dim : str
        The dimension name for samples in the dataset

    params : dict, optional
        Additional parameters for Affinity Propagation. Defaults include:
        - damping: 0.75
        - max_iter: 5000
        - convergence_iter: 250
        - affinity: 'precomputed'

    name : str, default='AffinityPropagation'
        The name to use when added to a Pipeline
    """

    def __init__(self, input_variable, output_variable, dim, params=None, name='AffinityPropagation'):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable, dim=dim,
                         params=params)

        default_params = {}
        default_params['damping'] = 0.75
        default_params['max_iter'] = 5000
        default_params['convergence_iter'] = 250
        default_params['affinity'] = 'precomputed'
        for k,v in default_params.items():
            if k not in self.params:
                self.params[k] = v

    def calculate(self, dataset):
        """Apply Affinity Propagation clustering to the dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing the similarity matrix

        Returns
        -------
        Self
            The AffinityPropagation instance with computed labels
        """
        data1 = self._get_variable(dataset)
        
        self.clf = sklearn.cluster.AffinityPropagation(**self.params)
        self.clf.fit(data1.values)
        
        self.labels = self.clf.labels_
        self.output[self.output_variable] = xr.DataArray(self.labels, dims=[self.dim])
        self.output[self.output_variable].attrs.update(self.params)
        self.output[self.output_variable].attrs['n_phases'] = len(np.unique(self.labels))
        return self
        
