import copy
from typing import Union, List, Optional

import xarray as xr

import AFL.double_agent.Pipeline

from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.util import listify

from AFL.automation.APIServer.Driver import Driver
from AFL.automation.shared.utilities import mpl_plot_to_bytes


class DoubleAgentDriver(Driver):
    defaults = {}
    defaults['data_path'] = '~/'
    defaults['AL_manifest_file'] = 'manifest.nc'
    defaults['save_path'] = '/home/AFL/'

    def __init__(self, overrides=None):
        Driver.__init__(self, name='SAS_AgentDriver', defaults=self.gather_defaults(), overrides=overrides)

        self.watchdog = None
        self._app = None
        self.name = 'DoubleAgentDriver'

        self.status_str = 'Fresh server!'

        self.datasets: Union[Optional[xr.Dataset], List[xr.Dataset]] = None
        self.pipelines: Union[Optional[Pipeline], List[Pipeline]] = None
        self.results: Union[Optional[xr.Dataset], List[xr.Dataset]] = None

    @property
    def app(self):
        return self._app

    @app.setter
    def app(self, value):
        self._app = value

    def predict(self):
        if (self.pipelines is None) or (self.datasets is None):
            raise ValueError(
                """Cannot predict without one or more pipelines loaded! Use client.set_driver_object to upload an """
                """Pipeline.Pipeline and an xr.Dataset to this APIServer. You currently have: \n"""
                f"""DoubleAgentDriver.pipelines = {self.pipelines}\n"""
                f"""DoubleAgentDriver.datasets = {self.datasets}\n"""
            )

        pipelines = listify(copy.deepcopy(self.pipelines))
        datasets = listify(copy.deepcopy(self.datasets))

        if len(datasets) == 1:
            datasets = datasets * len(pipelines)

        if not (len(pipelines) == len(datasets)):
            raise ValueError(
                """User must use client.set_driver_object to upload either a) one dataset or b) a number of datasets """
                """equal to the number of pipelines uploaded. You currently have: \n"""
                f"""len(DoubleAgentDriver.pipelines) = {len(listify(self.pipelines))}\n"""
                f"""len(DoubleAgentDriver.datasets) = {len(listify(self.datasets))}\n"""
            )

        results = []
        for dataset, pipeline in zip(datasets, pipelines):
            results.append(pipeline.calculate(dataset))


