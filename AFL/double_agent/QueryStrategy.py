from typing import List, Self
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp
import textwrap
from scipy.signal import find_peaks, peak_widths
from typing import Optional, Dict, Any

class ArgMax(PipelineOp):
    """
    Implements argmax query strategy for utility maximization.

    Parameters
    ----------
    input_variable : str, default="composition_utility"
        Dataset variable containing utility values.
    grid_variable : str, default="composition_marginal_domain"
        Dataset variable representing the design grid or domain.
    output_prefix : str, default="composition"
        Prefix for the output variable.
    name : str, default="ArgMax"
        Name of the pipeline operation.

    Methods
    -------
    calculate(dataset)
        Compute the point(s) in the grid that maximize the utility.
    """
    def __init__(
        self,
        input_variable: str = "composition_utility",
        grid_variable:str="composition_marginal_domain",
        output_prefix:str="composition",
        name: str = "ArgMax",
    ) -> None:
        output_variable = f"{output_prefix}_next"
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.grid_variable = grid_variable

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Compute the argmax of the utility over the design grid.

        If the grid is multidimensional, returns the full vector corresponding
        to the maximum utility.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the utility values and grid variable.

        Returns
        -------
        self : ArgMax
            Returns self with `output` containing the optimal point(s) that
            maximize the utility.
        """
        ux = dataset[self.input_variable]
        x = dataset[self.grid_variable]
        
        # x^* = argmax_{x} u(x)
        if x.ndim==1:
            x_opt = x.values[np.argmax(ux.values).item()]
            output = xr.DataArray(x_opt.reshape(-1, ),
                                dims=(self._prefix_output("next_n"), ),
                                )
        else:
            x_opt = x.values[np.argmax(ux.values).item(),:] 
            output = xr.DataArray(x_opt.reshape(-1, x.values.shape[-1]),
                                dims=(self._prefix_output("next_n"), x.dims[1]),
                                )

        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = textwrap.dedent("""
        Optimal composition to be added to the dataset
        that maximizes the utility.
        """).strip()
        return self

class FullWidthHalfMaximum1D(PipelineOp):
    """
    Find peaks in a 1D utility and compute their locations and
    full-width-half-maximum points, optionally including the global maximum.

    Parameters
    ----------
    input_variable : str, default="utility"
        Dataset variable containing 1D utility values.
    grid_variable : str, default="temperature_marginal_domain"
        Dataset variable representing the domain of the 1D function.
    output_variable : str, optional
        Name of the variable in the dataset to store the optimal points.
    params : dict, optional
        Extra keyword arguments to pass to `scipy.signal.find_peaks`.
    include_global_maximum : bool, default=True
        Include the location of the global utility maximum in the output.
    output_prefix : str, default="temperature"
        Prefix for the output variable.
    name : str, default="FullWidthHalfMaximum1D"
        Name of the pipeline operation.

    Methods
    -------
    calculate(dataset)
        Compute FWHM-based optimal locations and store them in the dataset.
    optimize(x, f, **kwargs)
        Identify peaks and their full-width-half-maximum locations in 1D data.
    """
    def __init__(
        self,
        input_variable: str = "utility",
        grid_variable:str="temperature_marginal_domain",
        output_variable: str = "next_sample",
        params: Optional[Dict[str, Any]] = None,
        include_global_maximum: bool = True,
        output_prefix:str="temperature",
        name: str = "FullWidthHalfMaximum1D",
    ) -> None:
        output_variable = f"{output_prefix}_next"
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.params = params or {}
        self.include_global_maximum = include_global_maximum
        self.grid_variable = grid_variable
    
    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Compute optimal 1D locations corresponding to full-width-half-maximum of peaks.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the 1D utility values and grid variable.

        Returns
        -------
        self : FullWidthHalfMaximum1D
            Returns self with `output` containing:
            
            - `output_variable`: Peak and FWHM locations for each peak and,
              by default, the global maximum location.
        """
        u = dataset[self.input_variable]
        x = dataset[self.grid_variable]

        x_opt = self.optimize(
            x.values,
            u.values,
            include_global_maximum=self.include_global_maximum,
            **self.params,
        )
        output = xr.DataArray(np.atleast_1d(x_opt.squeeze()),
                              dims=self._prefix_output("next_n"),
                            )
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = textwrap.dedent("""
        Optimal 1D locations at utility peaks and their full-width-half-maximum
        points, optionally including the global maximum.
        """).strip()
        return self

    def optimize(self, x, f, include_global_maximum=True, **kwargs):
        """
        Find peaks in `f(x)` and compute their full width at half maximum (FWHM).

        The global maximum location is included by default, even when it is not
        detected as a peak (for example, when it occurs at a domain boundary).

        Parameters
        ----------
        x : array-like
            1D array of x values.
        f : array-like
            1D array of function values.
        include_global_maximum : bool, default=True
            Whether to include the location of the global maximum.
        **kwargs : dict
            Extra keyword arguments for `scipy.signal.find_peaks`, e.g., height, distance, prominence.

        Returns
        -------
        xb : np.ndarray
            Sorted array of x values, rounded to two decimal places and unique
            at that precision. For each peak, its left half-maximum, peak, and
            right half-maximum locations are returned. The global maximum
            location is also returned when ``include_global_maximum`` is True.
        """
        x = np.asarray(x)
        f = np.asarray(f).squeeze()

        # Find peak indices
        peaks, _ = find_peaks(f, **kwargs)

        if len(peaks) > 0:
            # Compute widths at half maximum
            results_half = peak_widths(f, peaks, rel_height=0.5)

            # Left and right half-max positions 
            # (fractional indices, interpolate to x)
            left_ips = results_half[2]
            right_ips = results_half[3]

            xb = []
            for i, peak_idx in enumerate(peaks):
                left_x = np.interp(left_ips[i], np.arange(len(x)), x)
                peak_x = x[peak_idx]
                right_x = np.interp(right_ips[i], np.arange(len(x)), x)
                xb.extend([left_x, peak_x, right_x])

            xb = np.array(xb)
        else:
            xb = np.array([])

        if include_global_maximum:
            xb = np.append(xb, x[np.argmax(f)])

        return np.unique(np.round(xb, decimals=2))
    
class MinMax1DLineSampler(PipelineOp):
    """
    Generate line samples along a 1D domain based on feasibility probabilities.

    Parameters
    ----------
    input_variable : str, default="sliced_feasibility_y_prob"
        Dataset variable containing feasibility probabilities.
    grid_variable : str, default="temperature_domain"
        Dataset variable representing the 1D domain.
    min_value : float, default=0.3
        Minimum threshold for feasible probabilities.
    max_value : float, default=1.0
        Maximum threshold for feasible probabilities.
    n_samples : int, default=5
        Number of samples to generate along the feasible domain.
    output_prefix : str, default="temperature"
        Prefix for the output variable.
    name : str, default="TemperatureLineSampler"
        Name of the pipeline operation.

    Methods
    -------
    calculate(dataset)
        Generate line samples along the domain satisfying the probability constraints.
    linspace(arr)
        Return `n_samples` linearly spaced elements from an array.
    """
    def __init__(
        self,
        input_variable:str = "sliced_feasibility_y_prob",
        grid_variable:str = "temperature_domain",
        min_value: float = 0.3,
        max_value: float = 1.0,
        n_samples: int = 5,
        output_prefix:str="temperature",
        name: str = "TemperatureLineSampler",
    ) -> None:
        output_variable = f"{output_prefix}_next"
        super().__init__(
            name=name, 
            input_variable=[input_variable, grid_variable], 
            output_variable=output_variable
        )
        self.input_variable = input_variable
        self.grid_variable = grid_variable 
        self.min_value = min_value
        self.max_value = max_value
        self.n_samples = n_samples

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Generate feasible line samples based on probability thresholds.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the feasibility probabilities and 1D domain.

        Returns
        -------
        self : MinMax1DLineSampler
            Returns self with `output` containing:
            
            - `output_variable`: Array of linearly spaced feasible samples along the domain.
        """
        probs = dataset[self.input_variable]
        grid = dataset[self.grid_variable]
        flags = (probs>self.min_value) & (probs<self.max_value)
        feasible = grid[flags]

        line_samples = self.linspace(feasible)
        output = xr.DataArray(line_samples, 
                            dims=(self._prefix_output("next_n"), ),
                        )
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = textwrap.dedent("""
        Samples along the temperature axis that have higher probability of being feasible.
        The minimum is specified by the point that maximizes the acquisition
        function for determining phase boundaries (along with any associated costs)
        """).strip()

        return self 

    def linspace(self, arr):
        """
        Return n linearly spaced samples from an array.

        Parameters
        ----------
        arr : array-like
            Input array from which to sample.

        Returns
        -------
        samples : numpy.ndarray
            Array of n elements sampled evenly from arr.
        """
        arr = np.sort(arr)
        if self.n_samples <= 0:
            raise ValueError(f"self.n_samples={self.n_samples} must be positive")
        if self.n_samples > len(arr):
            raise ValueError(f"self.n_samples={self.n_samples} cannot be greater than the length of arr {arr.shape}")
        
        indices = np.linspace(0, len(arr) - 1, self.n_samples, dtype=int)
        return arr[indices]
