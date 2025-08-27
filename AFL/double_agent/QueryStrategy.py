from typing import List, Self
import numpy as np
import xarray as xr
from AFL.double_agent import PipelineOp
import textwrap
from scipy.signal import find_peaks, peak_widths
from typing import Optional, Dict, Any

class ArgMax(PipelineOp):
    """
    Query strategy that selects the design point with maximum utility.

    This class implements the argmax acquisition rule, identifying
    the point in the design space that maximizes the given utility
    function. It is particularly useful in Bayesian optimization or
    active learning where the next experiment is chosen based on
    maximizing a learned utility surface.

    Parameters
    ----------
    input_variable : str, default="composition_utility"
        Name of the dataset variable containing utility values.
    coordinate_dims : list of str, default=["protein", "glycerol"]
        Coordinate labels representing components of the design space.
    output_variable : str, default="next_sample"
        Name of the dataset variable where the optimal sample will be stored.
    name : str, default="ArgMax"
        Name of the pipeline operation.

    Attributes
    ----------
    coordinate_dims : list of str
        Coordinate labels used for the output representation.

    Methods
    -------
    calculate(dataset)
        Select the point with the highest utility and store it in the output.
    """
    def __init__(
        self,
        input_variable: str = "composition_utility",
        coordinate_dims= ["protein", "glycerol"],
        output_variable: str = "next_sample",
        name: str = "ArgMax",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.coordinate_dims = coordinate_dims

    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Identify the design point that maximizes the utility function.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing utility values and domain information.

        Returns
        -------
        ArgMax
            Updated instance with the optimal design point stored in `self.output`.
        """
        ux = dataset[self.input_variable]
        x = ux.domain
        x_opt = x[np.argmax(ux.values).item(),:] # x^* = argmax_{x} u(x)
        
        output = xr.DataArray(x_opt.reshape(-1, x.shape[-1]),
                              dims=("n_counts", "d_comp"),
                              coords={"d_comp": self.coordinate_dims}
                            )
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = textwrap.dedent("""
        Optimal composition to be added to the dataset
        that maximizes the utility.
        """).strip()
        return self

class FullWidthHalfMaximum1D(PipelineOp):
    """
    Query strategy that identifies full-width half-maximum (FWHM) points in 1D.

    This class finds peaks in a one-dimensional utility function and extracts
    the peak locations along with their left and right half-maximum points.
    If no peaks are detected, the global maximum is returned. This provides
    a way to select multiple informative candidates rather than just a single
    maximum.

    Parameters
    ----------
    input_variable : str, default="utility"
        Name of the dataset variable containing the 1D utility values.
    output_variable : str, default="next_sample"
        Name of the dataset variable where optimal sample locations will be stored.
    params : dict, optional
        Extra keyword arguments to pass to `scipy.signal.find_peaks`, such as
        `height`, `distance`, or `prominence` for controlling peak detection.
    name : str, default="FullWidthHalfMaximum1D"
        Name of the pipeline operation.

    Attributes
    ----------
    params : dict or None
        Parameters used for peak detection.

    Methods
    -------
    calculate(dataset)
        Find peaks in the utility and store corresponding FWHM points.
    optimize(x, f, **kwargs)
        Detect peaks and compute FWHM locations given a 1D function.
    """
    def __init__(
        self,
        input_variable: str = "utility",
        output_variable: str = "next_sample",
        params: Optional[Dict[str, Any]] = None,
        name: str = "FullWidthHalfMaximum1D",
    ) -> None:
        super().__init__(
            name=name, 
            input_variable=[input_variable], 
            output_variable=output_variable
        )
        self.params = params
    
    def calculate(self, dataset: xr.Dataset) -> Self:
        """
        Identify FWHM points in a 1D utility function.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing 1D utility values and domain information.

        Returns
        -------
        FullWidthHalfMaximum1D
            Updated instance with FWHM points stored in `self.output`.
        """
        u = dataset[self.input_variable]
        x = u.domain

        x_opt = self.optimize(x, u.values, **self.params)
        
        output = xr.DataArray(np.asarray(x_opt).reshape(-1),
                              dims=("n_next"),
                            )
        self.output[self.output_variable] = output
        self.output[self.output_variable].attrs["description"] = textwrap.dedent("""
        Optimal 1D locations that are the full-width-half-maximum points of the
        peaks in the utility.
        """).strip()
        return self

    def optimize(self, x, f, **kwargs):
        """
        Find peaks in a 1D function and compute their full-width half-maximum (FWHM).

        For each peak, returns three points: the left half-maximum location,
        the peak maximum, and the right half-maximum location. If no peaks
        are found, the function falls back to returning the global maximum.

        Parameters
        ----------
        x : array-like
            1D array of x values corresponding to the domain.
        f : array-like
            1D array of function values f(x).
        **kwargs : dict
            Extra keyword arguments passed to `scipy.signal.find_peaks`
            (e.g., `height=...`, `distance=...`, `prominence=...`).

        Returns
        -------
        ndarray
            Sorted array of x values. For each detected peak, three values
            are included: [left_half_max, peak_max, right_half_max].
            If no peaks are found, returns an array containing only
            the global maximum.
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

            xb = np.sort(np.array(xb))
        else:
            xb = np.array([x[np.argmax(f)]])  # fallback to global max

        return xb