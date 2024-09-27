import pathlib
import uuid
from typing import Optional, Dict, Any

import xarray as xr

from AFL.automation.APIServer.Driver import Driver  # type: ignore
from AFL.automation.shared.utilities import mpl_plot_to_bytes
from AFL.double_agent.Pipeline import Pipeline


class DoubleAgentDriver(Driver):
    """
    Persistent Config
    -----------------
    save_path: str
        path to directory where data will be serialized to
    """

    defaults = {}
    defaults["save_path"] = "/home/AFL/"

    def __init__(
        self,
        name: str = "DoubleAgentDriver",
        overrides: Optional[Dict[str, Any]] = None,
    ):
        Driver.__init__(
            self, name=name, defaults=self.gather_defaults(), overrides=overrides
        )
        self.app = None
        self.name = name

        self.status_str = "Fresh server!"

        self.input: Optional[xr.Dataset] = None
        self.pipeline: Optional[Pipeline] = None
        self.results: Dict[str, xr.Dataset] = dict()

    def initialize_input(self, db_uuid: str) -> None:
        """
        Set the initial input data to be evaluated in the `double_agent.Pipeline`

        Parameters
        ----------
        db_uuid: str
            Dropbox UUID to retrieve `xarray.Dataset` from. The Dataset should be deposited using `Client.deposit_obj`
            in interactive mode in order to obtain the uuid of the deposited item. See example below

        Example
        -------
        ```python
        from AFL.automation.APIServer.Client import Client
        import xarray as xr

        client = Client('localhost',port=5053)
        client.login('User')
        db_uuid = client.deposit_obj(obj=xr.Dataset(),interactive=True)['return_val']
        client.enqueue(task_name='initialize_input',db_uuid=db_uuid)
        ```
        """
        self.input = self.retrieve_obj(db_uuid)

    def initialize_pipeline(self, db_uuid: str) -> None:
        """
        Set the `double_agent.Pipeline` to outline

        Parameters
        ----------
        db_uuid: str
            Dropbox UUID to retrieve the `double_agent.Pipeline` from. The Dataset should be deposited using
            `Client.deposit_obj` in interactive mode in order to obtain the uuid of the deposited item. See example
            below

        Example
        -------
        ```python
        from AFL.automation.APIServer.Client import Client
        from AFL.double_agent import *

        client = Client('localhost',port=5053)
        client.login('User')
        db_uuid = client.deposit_obj(obj=Pipeline(),interactive=True)['return_val']
        client.enqueue(task_name='initialize_pipeline',db_uuid=db_uuid)
        ```
        """
        self.pipeline = self.retrieve_obj(db_uuid)

    def append(self, db_uuid: str, concat_dim: str) -> None:
        """

        Parameters
        ----------
        db_uuid: str
            Dropbox UUID to retrieve `xarray.Dataset` from. The Dataset should be deposited using `Client.deposit_obj`
            in interactive mode in order to obtain the uuid of the deposited item. See example below

        concat_dim: str
            `xarray` dimension in input dataset to concatenate to

        """
        if self.input is None:
            raise ValueError(
                'Must set "input" Dataset client.deposit_obj and then DoubleAgentDriver.initialize'
            )

        next_sample = self.retrieve_obj(db_uuid)

        if self.input is None:
            self.input = next_sample
        else:
            self.input = xr.concat(
                [self.input, next_sample], dim=concat_dim, data_vars="minimal"
            )

    @Driver.unqueued(render_hint = 'precomposed_svg')
    def plot_pipeline(self,**kwargs):
        if self.pipeline is not None:
            return mpl_plot_to_bytes(self.pipeline.draw(),format='svg')
        else:
            return None

    @Driver.unqueued(render_hint = 'html')
    def last_results(self,**kwargs):
        return self.last_results._repr_html_()

    @Driver.unqueued(render_hint = 'precomposed_png')
    def plot_operation(self,operation,**kwargs):
        try:
            operation = int(operation)
        except ValueError:
            pass
        if self.pipeline is not None:
            if isinstance(operation,str):
                return mpl_plot_to_bytes(self.pipeline.search(operation).plot(),format='png')
            elif isinstance(operation,int):
                return mpl_plot_to_bytes(self.pipeline[operation].plot(),format='png')
            else:
                return None
        else:
            return None
        
    def reset_results(self):
        self.results = dict()

    def predict(
        self,
        deposit: bool = True,
        save_to_disk: bool = True,
        sample_uuid: Optional[str] = None,
        AL_campaign_name: Optional[str] = None,
    ) -> str:
        """
        Evaluate the pipeline set with `.initialize_pipeline`.

        Parameters
        ----------
        deposit: bool
            If True, the `xarray.Dataset` resulting from the `Pipeline` calculation will be placed in this `APIServers`
            dropbox for retrieval and the `db_uuid` will be returned.

        save_to_disk: bool
            If True, the `xarray.Dataset` resulting from the `Pipeline` calculation will be serialized to disk in
            NetCDF format.

        sample_uuid: Optional[str]
            Optionally provide a sample uuid to tag the calculation with

        AL_campaign_name
            Optionally provide an AL campaign name to tag the calculation with

        """
        if (self.pipeline is None) or (self.input is None):
            raise ValueError(
                """Cannot predict without a pipeline and input loaded! Use client.set_driver_object to upload an """
                """Pipeline.Pipeline and an xr.Dataset to this APIServer. You currently have: \n"""
                f"""DoubleAgentDriver.pipeline = {self.pipeline}\n"""
                f"""DoubleAgentDriver.input = {self.input}\n"""
            )

        ag_uid = "AG-" + str(uuid.uuid4())
        self.results[ag_uid] = self.pipeline.calculate(self.input)

        self.results[ag_uid].attrs['sample_uuid'] = sample_uuid
        self.results[ag_uid].attrs['ag_uuid'] = ag_uid
        self.results[ag_uid].attrs['AL_campaign_name'] = AL_campaign_name

        if save_to_disk:
            path = (
                pathlib.Path(self.config["save_path"])
                / f"{AL_campaign_name}_SAM-{str(sample_uuid)[-6:]}_AG-{ag_uid[-6:]}.nc"
            )
            self.results[ag_uid].to_netcdf(path)

        if deposit:
            self.deposit_obj(self.results[ag_uid], uid=ag_uid)
        
        self.last_results = self.results[ag_uid]
        
        return ag_uid

_OVERRIDE_MAIN_MODULE_NAME = 'DoubleAgentDriver'
if __name__ == '__main__':
    from AFL.automation.shared.launcher import *
