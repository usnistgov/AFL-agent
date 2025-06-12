from typing import Optional,  Dict, Any
import warnings 

import numpy as np
import scipy.spatial  # type: ignore
import xarray as xr
from typing_extensions import Self

from AFL.double_agent.PairMetric import PairMetric
from scipy.interpolate import UnivariateSpline

try:
    from apdist.torch import TorchAmplitudePhaseDistance as apdist 
    import torch
except ImportError:
    raise RuntimeError((
    "To use amplitude-distance as a similarity measure, please install: "
    "pip install git+https://github.com/kiranvad/Amplitude-Phase-Distance"
    ))

class AmplitudePhaseDistance(PairMetric):
    """Computes pairwise amplitude phase distance between samples
    
    The amplitude-phase distance is a measure of shape between a pair of 
    spectra-like measurement curves (i.e. SAXS, UV-Vis, XRD) using a differential
    geometry method. It represents spectra as a one-dimensional function and measures
    pairwise distance by computing the shape mis-match on x and y-axis.

    In simple terms, Amplitude measures changes to shape on the y-axis
    of a 1D function and Phase measures changes to the shape on x-axis.

    Vaddi, K., Chiang, H. T., & Pozzo, L. D. "Autonomous retrosynthesis of gold 
    nanoparticles via spectral shape matching." Digital Discovery, vol. 1, no. 4 
    (2022): 502-510. Royal Society of Chemistry.
    
    Original paper URL: https://pubs.rsc.org/en/content/articlehtml/2022/dd/d2dd00025c

    Parameters
    ----------
    input_variable : str
        The name of the variable on which to apply this PairMetric   
    output_variable : str
        The name of the variable to be inserted into the dataset
    sample_dim : str
        The dimension containing different samples
    params : Dict
        Parameters for the distance function.
        See https://github.com/kiranvad/funcshape/funcshape/functions.py#L157
        alpha : float, default=0.5
            An additional parameter that weighs amplitude and phase contrirubutions.
    name : str, default="AmplitudePhaseDistance"
        The name to use when added to a Pipeline
    """    
    def __init__(
        self,
        input_variable : str,
        output_variable: str,
        sample_dim: str,
        params: Optional[Dict[str, Any]] = None,
        name="AmplitudePhaseDistance",
    ) -> None:
        super().__init__(
            name=name,
            input_variable=input_variable,
            output_variable=output_variable,
            sample_dim=sample_dim,
            params=params
        )


    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`"""
        data = self._get_variable(dataset)

        domain_variable = [d for d in data.dims if d != self.sample_dim][0]
        domain = data.coords[domain_variable].values
        codomain = data.values

        # use log10 transformation to amplify functional signals (e.g.: peaks)
        metric = lambda y1, y2 : self._get_pairiwise_ap(domain, y1, y2)

        # similarity matrix of length sample x sample
        pair_dists = scipy.spatial.distance.pdist(codomain, metric=metric)
        self.W = scipy.spatial.distance.squareform(pair_dists)

        dims = [self.sample_dim + "_i", self.sample_dim + "_j"]
        self.output[self.output_variable] = xr.DataArray(self.W, dims=dims)  # type: ignore
        self.output[self.output_variable].attrs.update(self.params)  # type: ignore

        return self 
    
    def _get_pairiwise_ap(self, t, f_ref, f_query):
        optim_kwargs = {"n_iters":100, 
                "n_basis":20, 
                "n_layers":15,
                "domain_type":"linear",
                "basis_type":"palais",
                "n_restarts":50,
                "lr":1e-1,
                "n_domain":len(t)
            }
        optim_kwargs.update(self.params)
        alpha = optim_kwargs.get("alpha", 0.5)

        xs = np.linspace(min(t), max(t), len(t))
        spl = UnivariateSpline(t, f_ref)
        ys_ref = spl(xs)
        spl = UnivariateSpline(t, f_query)
        ys_query = spl(xs)        

        amplitude, phase, _ = apdist(torch.from_numpy(xs),
                                     torch.from_numpy(ys_ref),
                                     torch.from_numpy(ys_query), 
                                     **optim_kwargs
                                )
        return alpha*amplitude + (1-alpha)*phase
