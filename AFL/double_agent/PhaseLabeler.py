import copy
import warnings
from collections import Counter
from collections import defaultdict

import numpy as np
import xarray as xr
import sklearn.cluster
import sklearn.mixture
from sklearn.metrics import silhouette_samples

from AFL.double_agent.Pipeline import PipelineOpBase


class PhaseLabeler(PipelineOpBase):
    def __init__(self, input_variable, output_variable, dim='sample', params=None, name='PhaseLabeler'):

        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.labels = None
        self.dim = dim

        if params is None:
            self.params = {}
        else:
            self.params = params

        self._banned_from_attrs.append('labels')

    def __getitem__(self, index):
        return self.labels[index]

    def __array__(self, dtype=None):
        return np.array(self.labels).astype(dtype)

    def remap_labels_by_count(self):
        label_map = {}
        for new_label, (old_label, _) in enumerate(
                sorted(Counter(self.labels).items(), key=lambda x: x[1], reverse=True)):
            label_map[old_label] = new_label
        self.labels = list(map(label_map.get, self.labels))

    def silhouette(self, W):
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
            # silh_dict['labelers'].append(self.copy())

        silh_avg = np.array(silh_dict['avg_scores'])
        found = False
        for cutoff in np.arange(0.85, 0.4, -0.05):
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


class SpectralClustering(PhaseLabeler):
    def __init__(self, input_variable, output_variable, dim, params=None, name='SpectralClustering'):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable, dim=dim,
                         params=params)

    def _construct_labeler(self, **params):
        self.params.update(params)

        self.clf = sklearn.cluster.SpectralClustering(
            self.params['n_phases'],
            affinity='precomputed',
            assign_labels="discretize",
            random_state=0,
            n_init=1000
        )

    def calculate(self, dataset):
        data1 = self._get_variable(dataset)
        self._construct_labeler()
        self.silhouette(data1.values)  # maybe silhouette should be a separate op...
        self.output[self.output_variable] = xr.DataArray(self.labels, dims=[self.dim])
        return self

    def _label(self, metric):
        """Hidden method for actually doing the labeler fit"""
        self.clf.fit(metric)
        self.labels = self.clf.labels_


class GaussianMixtureModel(PhaseLabeler):
    def __init__(self, input_variable, output_variable, dim, params=None, name='GaussianMixtureModel'):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable, dim=dim,
                         params=params)

    def _construct_labeler(self, **params):
        self.params.update(params)
        self.clf = sklearn.mixture.GaussianMixture(self.params['n_phases'])

    def calculate(self, dataset):
        data1 = self._get_variable(dataset)
        self._construct_labeler()
        self.silhouette(data1.values)
        self.output[self.output_variable] = xr.DataArray(self.labels, dims=[self.dim])
        return self

    def _label(self, metric):
        self.clf.fit(metric)
        self.labels = self.clf.predict(metric)
