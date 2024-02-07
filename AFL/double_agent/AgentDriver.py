from typing import Union, Optional, Dict
import uuid

import xarray as xr

from AFL.double_agent.Pipeline import Pipeline
from AFL.automation.APIServer.Driver import Driver


class DoubleAgentDriver(Driver):
    defaults = {}
    defaults['save_path'] = '/home/AFL/'

    def __init__(self, overrides=None):
        Driver.__init__(self, name='SAS_AgentDriver', defaults=self.gather_defaults(), overrides=overrides)

        self.watchdog = None
        self.app = None
        self.name = 'DoubleAgentDriver'

        self.status_str = 'Fresh server!'

        self.input: Optional[xr.Dataset] = None
        self.pipeline: Optional[Pipeline] = None
        self.results: Optional[Dict[str, xr.Dataset]] = None

    def predict(self, deposit=True):
        if (self.pipeline is None) or (self.input is None):
            raise ValueError(
                """Cannot predict without one or more pipelines loaded! Use client.set_driver_object to upload an """
                """Pipeline.Pipeline and an xr.Dataset to this APIServer. You currently have: \n"""
                f"""DoubleAgentDriver.pipelines = {self.pipeline}\n"""
                f"""DoubleAgentDriver.datasets = {self.input}\n"""
            )

        al_uid = 'AL-' + str(uuid.uuid4())
        self.results[al_uid] = self.pipeline.calculate(self.input)

        if deposit:
            self.deposit_to_dropbox(self.results[al_uid], uid=al_uid)

    def deposit_to_dropbox(self, obj: object, uid: Union[str, uuid.UUID]) -> None:
        if uid is None:
            uid = 'DB-' + str(uuid.uuid4())

        self.dropbox = {}
        self.dropbox[uid] = obj
