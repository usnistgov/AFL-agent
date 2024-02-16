from typing import Union, Optional, Dict
import uuid
import pathlib

import xarray as xr

from AFL.double_agent.Pipeline import Pipeline
from AFL.automation.APIServer.Driver import Driver


class DoubleAgentDriver(Driver):
    defaults = {}
    defaults['save_path'] = '/home/AFL/'

    def __init__(self, name='DoubleAgentDriver', overrides=None):
        Driver.__init__(self, name=name, defaults=self.gather_defaults(), overrides=overrides)
        self.app = None
        self.name = name

        self.status_str = 'Fresh server!'

        self.input: Optional[xr.Dataset] = None
        self.pipeline: Optional[Pipeline] = None
        self.results: Optional[Dict[str, xr.Dataset]] = dict()

        
    def initialize_input(self,db_uuid:str):
        self.input = self.retrieve_obj(db_uuid)
        
    def initialize_pipeline(self,db_uuid:str):
        self.pipeline = self.retrieve_obj(db_uuid)
        
    def append(self,db_uuid:str,concat_dim:str):
        if self.input is None:
            raise ValueError('Must set "input" Dataset client.deposit_obj and then DoubleAgentDriver.initialize')
                             
        next_sample = self.retrieve_obj(db_uuid)
              
        self.input = xr.concat([self.input,next_sample],dim=concat_dim,data_vars='minimal')
        
    def reset_results(self):
        self.results = dict()

    def predict(self, deposit=True, save_to_disk=True, sample_uuid=None, AL_campaign_name=None):
        if (self.pipeline is None) or (self.input is None):
            raise ValueError(
                """Cannot predict without a pipeline and input loaded! Use client.set_driver_object to upload an """
                """Pipeline.Pipeline and an xr.Dataset to this APIServer. You currently have: \n"""
                f"""DoubleAgentDriver.pipeline = {self.pipeline}\n"""
                f"""DoubleAgentDriver.input = {self.input}\n"""
            )

        ag_uid = 'AG-' + str(uuid.uuid4())
        self.results[ag_uid] = self.pipeline.calculate(self.input)

        if save_to_disk:
            path = (
                    pathlib.Path(self.config['save_path']) /
                    f'{AL_campaign_name}_SAM-{str(sample_uuid)[-6:]}_AG-{ag_uid[-6:]}.nc'
            )
            self.results[ag_uid].to_netcdf(path)

        if deposit:
            self.deposit_obj(self.results[ag_uid],uid=ag_uid)

        return ag_uid