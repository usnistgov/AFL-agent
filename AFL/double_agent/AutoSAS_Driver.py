import copy
import pathlib
import uuid
from collections import defaultdict
from typing import Dict, Any, Optional, List, Union

import h5py  # type: ignore
import numpy as np
import pandas as pd
import xarray as xr

from AFL.automation.APIServer.Driver import Driver  # type: ignore
from AFL.double_agent.AutoSAS import SASModel, SASFitter


class AutoSAS_Driver(Driver):
    defaults = {}
    defaults["savepath"] = "/home/afl642/2402_DT_ISIS_path"
    defaults["q_min"] = 1e-2
    defaults["q_max"] = 1e-1
    defaults["resolution"] = None
    defaults["model_inputs"] = SASFitter.DEFAULT_MODEL_INPUTS
    defaults["fit_method"] = SASFitter.DEFAULT_FIT_METHOD

    def __init__(self):
        Driver.__init__(self, name="SAS_model_fitter", defaults=self.gather_defaults())
        self.status_str = "Fresh Server!"
        self.dropbox = dict()
        self.fitter = None
        print("self.data exists == :", self.data)

    def status(self):
        status = []
        status.append(self.status_str)
        return status

    def update_status(self, status):
        self.status_str = status
        if self.app is not None:
            self.app.logger(status)

    def set_sasdata(
        self,
        db_uuid: str,
        sample_dim: str = "sample",
        q_variable: str = "q",
        sas_variable: str = "I",
        sas_err_variable: str = "dI",
        sas_resolution_variable: Optional[str] = None,
    ) -> None:
        """
        Set the sasdata to be fit from a dropbox UUID
        
        Parameters
        ----------
        db_uuid: str
            Dropbox UUID to retrieve `xarray.Dataset` from
            
        sample_dim: str
            The `xarray` dimension containing each sample

        q_variable: str
            The name of the `xarray.Dataset` variable corresponding to the q values

        sas_variable: str
            The name of the `xarray.Dataset` variable corresponding to the scattering intensity to be fit

        sas_err_variable: str
            The name of the `xarray.Dataset` variable corresponding to the uncertainty in the scattering intensity

        sas_resolution_variable: Optional[str]
            The name of the `xarray.Dataset` variable corresponding to the resolution function
        """
        dataset = self.retrieve_obj(db_uuid)
        
        # Initialize the fitter
        self.fitter = SASFitter(
            model_inputs=self.config["model_inputs"],
            fit_method=self.config["fit_method"],
            q_min=self.config["q_min"],
            q_max=self.config["q_max"],
            resolution=self.config["resolution"]
        )
        
        # Set the SAS data in the fitter
        self.fitter.set_sasdata(
            dataset=dataset,
            sample_dim=sample_dim,
            q_variable=q_variable,
            sas_variable=sas_variable,
            sas_err_variable=sas_err_variable,
            sas_resolution_variable=sas_resolution_variable
        )

    def fit_models(self, parallel=False, fit_method=None):
        """
        Execute SAS model fitting using the fitter
        
        Parameters
        ----------
        parallel: bool
            NOT IMPLEMENTED! Flag for parallel processing
            
        fit_method: Optional[Dict]
            Custom fit method parameters
            
        Returns
        -------
        str:
            UUID of the fit calculation
        """
        if self.fitter is None:
            raise ValueError("No SAS data set. Use set_sasdata first.")
        
        # Update fit method if provided
        if fit_method is not None:
            self.config["fit_method"] = fit_method
            self.fitter.fit_method = fit_method
        
        # Perform the fitting
        fit_uuid, output_dataset = self.fitter.fit_models(parallel=parallel)
        
        # Store in data and deposit to dropbox
        for name, array in output_dataset.data_vars.items():
            self.data.add_array(name, array.values)
            
            if array.dims:
                for dim in array.dims:
                    if dim in output_dataset.coords:
                        dim_name = f"{name}_dim_{dim}"
                        self.data.add_array(dim_name, output_dataset[dim].values)
        
        # Store best fit info
        self.data.add_array("best_chisq", self.fitter.report["best_fits"]["lowest_chisq"])
        self.data.add_array("model_names", self.fitter.report["best_fits"]["model_name"])
        self.data.add_array("all_chisq", self.fitter.report["all_chisq"])
        self.data.add_array("probabilities", self.fitter.report["probabilities"])
        
        # Deposit the output dataset to the dropbox
        self.deposit_obj(obj=output_dataset, uid=fit_uuid)
        
        return fit_uuid

    def _writedata(self, data):
        filename = pathlib.Path(self.config["filename"])
        filepath = pathlib.Path(self.config["filepath"])
        print(f"writing data to {filepath/filename}")
        with h5py.File(filepath / filename, "w") as f:
            f.create_dataset(str(uuid.uuid1()), data=data)

_DEFAULT_PORT = 5058
if __name__ == '__main__':
    from AFL.automation.shared.launcher import *
