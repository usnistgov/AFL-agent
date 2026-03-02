import copy
import datetime
import json
import pathlib
import uuid
from collections import defaultdict
from typing import Dict, Any, Optional, List, Union

import h5py  # type: ignore
import numpy as np
import pandas as pd
import xarray as xr
from tiled.queries import Eq  # type: ignore

from AFL.automation.APIServer.Driver import Driver  # type: ignore
from AFL.double_agent.AutoSAS import SASModel, SASFitter
from AFL.double_agent.AutoSASWebAppMixin import AutoSASWebAppMixin


class AutoSAS_Driver(AutoSASWebAppMixin, Driver):
    defaults = {}
    defaults["savepath"] = "/home/afl642/2402_DT_ISIS_path"
    defaults["q_min"] = 1e-2
    defaults["q_max"] = 1e-1
    defaults["resolution"] = None
    defaults["model_inputs"] = SASFitter.DEFAULT_MODEL_INPUTS
    defaults["fit_method"] = SASFitter.DEFAULT_FIT_METHOD

    def __init__(self):
        Driver.__init__(self, name="SAS_model_fitter", defaults=self.gather_defaults())
        self._autosas_webapp_init()
        self.status_str = "Fresh Server!"
        self.fitter = None
        self._autosas_input_dataset = None
        self._autosas_sample_dim = "sample"
        self._autosas_q_variable = "q"
        self._autosas_sas_variable = "I"
        self._autosas_sas_err_variable = "dI"
        self._autosas_sas_resolution_variable = None
        print("self.data exists == :", self.data)

    @property
    def tiled_client(self):
        if self.data is None:
            raise ValueError("No data object found on driver; cannot access tiled_client.")
        client = getattr(self.data, "tiled_client", None)
        if client is None:
            raise ValueError("self.data does not expose tiled_client.")
        return client

    @staticmethod
    def _coerce_dataset_dict(dataset_dict: Any) -> xr.Dataset:
        if dataset_dict is None:
            raise ValueError("dataset_dict is required when entry_ids are not provided.")
        if isinstance(dataset_dict, str):
            dataset_dict = json.loads(dataset_dict)
        if not isinstance(dataset_dict, dict):
            raise ValueError("dataset_dict must be a dict or JSON string containing xr.Dataset.to_dict output.")
        try:
            dataset = xr.Dataset.from_dict(dataset_dict)
        except Exception as exc:
            raise ValueError(f"Could not reconstruct dataset from dataset_dict: {exc}") from exc
        return dataset

    @staticmethod
    def _parse_entry_ids(entry_ids: Any) -> List[str]:
        if entry_ids is None:
            return []
        if isinstance(entry_ids, str):
            stripped = entry_ids.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                parsed = json.loads(stripped)
            else:
                parsed = [token.strip() for token in stripped.split(",")]
        elif isinstance(entry_ids, (list, tuple)):
            parsed = list(entry_ids)
        else:
            raise ValueError("entry_ids must be a list/tuple or a JSON/CSV string.")

        normalized: List[str] = []
        for val in parsed:
            token = str(val).strip()
            if token:
                normalized.append(token)
        return normalized

    def _read_tiled_item(self, item: Any):
        try:
            return item.read(optimize_wide_table=False)
        except TypeError as exc:
            if "optimize_wide_table" in str(exc) or "unexpected keyword" in str(exc):
                return item.read()
            raise

    @staticmethod
    def _get_nested_metadata_value(metadata: Dict[str, Any], path: str) -> Any:
        current: Any = metadata
        for part in path.split("."):
            if not isinstance(current, dict):
                return None
            if part not in current:
                return None
            current = current[part]
        return current

    def _entry_sort_timestamp(self, entry: Any) -> datetime.datetime:
        metadata = dict(getattr(entry, "metadata", {}) or {})
        timestamp_paths = (
            "meta.ended",
            "attrs.meta.ended",
            "attr.meta.ended",
            "meta.started",
            "attrs.meta.started",
            "attr.meta.started",
            "timestamp",
            "attrs.timestamp",
            "attr.timestamp",
        )
        for path in timestamp_paths:
            value = self._get_nested_metadata_value(metadata, path)
            if isinstance(value, datetime.datetime):
                return value
            if isinstance(value, str):
                try:
                    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    continue
        return datetime.datetime.min

    def _iter_fit_entries(
        self,
        fit_uuid: Optional[str],
        task_name: str,
        allow_task_fallback: bool = True,
    ) -> List[tuple[str, Any]]:
        task_fields = ("task_name", "attrs.task_name", "attr.task_name")
        fit_uuid_fields = ("attrs.fit_uuid", "attr.fit_uuid", "fit_uuid")

        entries: Dict[str, Any] = {}
        query_errors: List[str] = []

        for task_field in task_fields:
            try:
                base = self.tiled_client.search(Eq(task_field, task_name))
            except Exception as exc:
                query_errors.append(str(exc))
                continue

            if fit_uuid is None:
                try:
                    for entry_id, entry in base.items():
                        entries[entry_id] = entry
                except Exception as exc:
                    query_errors.append(str(exc))
                continue

            matched_for_fit = False
            for fit_field in fit_uuid_fields:
                try:
                    for entry_id, entry in base.search(Eq(fit_field, fit_uuid)).items():
                        entries[entry_id] = entry
                        matched_for_fit = True
                except Exception as exc:
                    query_errors.append(str(exc))
                    continue
            if matched_for_fit:
                break

        if entries:
            return list(entries.items())

        if fit_uuid is not None and allow_task_fallback:
            # Some deployments may not index fit_uuid metadata; fall back to task-level latest lookup.
            return self._iter_fit_entries(
                fit_uuid=None,
                task_name=task_name,
                allow_task_fallback=allow_task_fallback,
            )

        if fit_uuid is None:
            raise ValueError(
                f"No tiled fit entries found for task_name='{task_name}'. Query errors: {' | '.join(query_errors)}"
            )
        raise ValueError(
            f"No tiled fit entry found for fit_uuid='{fit_uuid}' and task_name='{task_name}'. "
            f"Query errors: {' | '.join(query_errors)}"
        )

    def _select_latest_entry(self, entries: List[tuple[str, Any]]) -> tuple[str, Any]:
        indexed = list(enumerate(entries))
        latest = max(
            indexed,
            key=lambda indexed_item: (self._entry_sort_timestamp(indexed_item[1][1]), indexed_item[0]),
        )
        return latest[1]

    def _annotate_output_dataset(self, dataset: xr.Dataset, fit_uuid: str, task_name: str) -> xr.Dataset:
        output = dataset.copy(deep=True)
        now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
        output.attrs.update(
            {
                "fit_uuid": fit_uuid,
                "task_name": task_name,
                "fit_method": copy.deepcopy(self.config.get("fit_method")),
                "model_inputs": copy.deepcopy(self.config.get("model_inputs")),
                "q_min": self.config.get("q_min"),
                "q_max": self.config.get("q_max"),
                "resolution": self.config.get("resolution"),
                "timestamp": now,
            }
        )
        for var_name in output.data_vars:
            output[var_name].attrs["fit_uuid"] = fit_uuid
            output[var_name].attrs["task_name"] = task_name
            output[var_name].attrs["timestamp"] = now
        return output

    def _build_fitted_input_dataset(self) -> xr.Dataset:
        """Build a dataset view of the raw SAS data used in the fit."""
        target_sample_dim = "sas_fit_sample"
        q_dim_name = "sas_input_q"

        def _align_q_coordinate(ds: xr.Dataset) -> xr.Dataset:
            """Set sas_input_q coordinate values equal to the q variable values."""
            if "q" not in ds or q_dim_name not in ds.dims:
                return ds
            q_da = ds["q"]
            if q_dim_name not in q_da.dims:
                return ds
            other_dims = [dim for dim in q_da.dims if dim != q_dim_name]
            if other_dims:
                selector = {dim: 0 for dim in other_dims}
                q_coord_vals = np.asarray(q_da.isel(**selector).values)
            else:
                q_coord_vals = np.asarray(q_da.values)
            return ds.assign_coords({q_dim_name: q_coord_vals})

        # Primary path: pull from the loaded xarray dataset so we preserve original structure.
        if self._autosas_input_dataset is not None:
            source = self._autosas_input_dataset.copy(deep=True)
            sample_dim = self._autosas_sample_dim or "sample"
            if sample_dim in source.dims and sample_dim != target_sample_dim:
                source = source.rename_dims({sample_dim: target_sample_dim})
            if sample_dim in source.coords and sample_dim != target_sample_dim:
                source = source.rename_vars({sample_dim: target_sample_dim})

            variables_to_include = [
                self._autosas_q_variable,
                self._autosas_sas_variable,
                self._autosas_sas_err_variable,
            ]
            if self._autosas_sas_resolution_variable:
                variables_to_include.append(self._autosas_sas_resolution_variable)

            input_dataset = xr.Dataset()
            for var_name in variables_to_include:
                if not var_name or var_name not in source:
                    continue
                input_dataset[var_name] = source[var_name].copy(deep=True)
                input_dataset[var_name].attrs["source_variable"] = var_name
                # Also expose stable aliases expected downstream.
                if var_name == self._autosas_q_variable and "q" not in input_dataset:
                    input_dataset["q"] = source[var_name].copy(deep=True)
                if var_name == self._autosas_sas_variable and "I" not in input_dataset:
                    input_dataset["I"] = source[var_name].copy(deep=True)
                if var_name == self._autosas_sas_err_variable and "dI" not in input_dataset:
                    input_dataset["dI"] = source[var_name].copy(deep=True)

            if len(input_dataset.data_vars) > 0:
                return _align_q_coordinate(input_dataset)

        # Fallback path: construct directly from fitter.sasdata so q/I/dI are always present.
        if self.fitter is None or not getattr(self.fitter, "sasdata", None):
            return xr.Dataset()

        sasdata = list(self.fitter.sasdata)
        n_samples = len(sasdata)
        max_len = max(len(np.asarray(d.x)) for d in sasdata)
        q_arr = np.full((n_samples, max_len), np.nan, dtype=float)
        i_arr = np.full((n_samples, max_len), np.nan, dtype=float)
        di_arr = np.full((n_samples, max_len), np.nan, dtype=float)
        dq_arr = None
        has_dx = any(getattr(d, "dx", None) is not None for d in sasdata)
        if has_dx:
            dq_arr = np.full((n_samples, max_len), np.nan, dtype=float)

        for idx, data in enumerate(sasdata):
            x = np.asarray(data.x, dtype=float)
            y = np.asarray(data.y, dtype=float)
            dy = np.asarray(data.dy, dtype=float)
            n = len(x)
            q_arr[idx, :n] = x
            i_arr[idx, :n] = y
            di_arr[idx, :n] = dy
            dx = getattr(data, "dx", None)
            if dq_arr is not None and dx is not None:
                dq_arr[idx, :n] = np.asarray(dx, dtype=float)

        input_dataset = xr.Dataset(
            {
                "q": xr.DataArray(q_arr, dims=[target_sample_dim, q_dim_name]),
                "I": xr.DataArray(i_arr, dims=[target_sample_dim, q_dim_name]),
                "dI": xr.DataArray(di_arr, dims=[target_sample_dim, q_dim_name]),
            },
            coords={target_sample_dim: np.arange(n_samples), q_dim_name: q_arr[0, :]},
        )
        if dq_arr is not None:
            input_dataset["dq"] = xr.DataArray(dq_arr, dims=[target_sample_dim, q_dim_name])
        return _align_q_coordinate(input_dataset)

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
        db_uuid: Optional[str] = None,
        entry_ids: Any = None,
        dataset_dict: Any = None,
        concat_dim: Optional[str] = None,
        variable_prefix: str = "",
        sample_dim: str = "sample",
        q_variable: str = "q",
        sas_variable: str = "I",
        sas_err_variable: str = "dI",
        sas_resolution_variable: Optional[str] = None,
    ) -> None:
        """
        Set the sasdata to be fit from Tiled entry_ids or an xr.Dataset dict payload.
        
        Parameters
        ----------
        db_uuid: Optional[str]
            Legacy compatibility input; when provided without entry_ids, it is treated as
            a single Tiled entry id.
        entry_ids: Any
            Tiled entry id list (list/tuple), CSV string, or JSON list string.
        dataset_dict: Any
            `xr.Dataset.to_dict()` payload (dict or JSON string) for server-side ingest.
            
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
        parsed_entry_ids = self._parse_entry_ids(entry_ids)
        if not parsed_entry_ids and db_uuid:
            parsed_entry_ids = [str(db_uuid)]
        if parsed_entry_ids:
            dataset = self.tiled_concat_datasets(
                entry_ids=parsed_entry_ids,
                concat_dim=concat_dim or sample_dim,
                variable_prefix=variable_prefix or "",
            ).reset_coords()
        elif dataset_dict is not None:
            dataset = self._coerce_dataset_dict(dataset_dict).reset_coords()
        else:
            raise ValueError(
                "AutoSAS set_sasdata now requires Tiled-backed input: provide entry_ids "
                "or dataset_dict (xr.Dataset.to_dict payload)."
            )

        self._autosas_input_dataset = dataset
        self._autosas_sample_dim = sample_dim
        self._autosas_q_variable = q_variable
        self._autosas_sas_variable = sas_variable
        self._autosas_sas_err_variable = sas_err_variable
        self._autosas_sas_resolution_variable = sas_resolution_variable
        
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
        xr.Dataset
            Fit results dataset with fit metadata stored in attrs.
        """
        if self.fitter is None:
            raise ValueError("No SAS data set. Use set_sasdata first.")
        
        # Update fit method if provided
        if fit_method is not None:
            self.config["fit_method"] = fit_method
            self.fitter.fit_method = fit_method
        
        # Perform the fitting and construct a fully self-contained dataset result.
        fit_uuid, output_dataset = self.fitter.fit_models(parallel=parallel)
        output_dataset = output_dataset.copy(deep=True)
        output_dataset["best_chisq"] = xr.DataArray(
            data=np.asarray(self.fitter.report["best_fits"]["lowest_chisq"]),
            dims=["sas_fit_sample"],
            coords={"sas_fit_sample": output_dataset.coords["sas_fit_sample"]},
        )
        output_dataset["model_names"] = xr.DataArray(
            data=np.asarray(self.fitter.report["best_fits"]["model_name"]),
            dims=["sas_fit_sample"],
            coords={"sas_fit_sample": output_dataset.coords["sas_fit_sample"]},
        )
        input_dataset = self._build_fitted_input_dataset()
        if len(input_dataset.data_vars) > 0:
            output_dataset = xr.merge([output_dataset, input_dataset], compat="override")
        output_dataset = self._annotate_output_dataset(
            dataset=output_dataset,
            fit_uuid=fit_uuid,
            task_name="fit_models",
        )

        self._last_fit_uuid = fit_uuid
        self._last_fit_dataset = output_dataset

        return output_dataset

    @Driver.unqueued()
    def get_fit_dataset(self, fit_uuid: Optional[str] = None, fit_task_name: str = "fit_models", **kwargs):
        """Return a fit result dataset from Tiled, defaulting to the latest fit task."""
        metadata: Dict[str, Any] = {}
        try:
            entries = self._iter_fit_entries(
                fit_uuid=fit_uuid,
                task_name=fit_task_name,
                allow_task_fallback=True,
            )
            entry_id, entry = self._select_latest_entry(entries)
            dataset = self._read_tiled_item(entry)
            if not isinstance(dataset, xr.Dataset):
                raise ValueError(
                    f"Tiled fit entry '{entry_id}' did not return an xarray.Dataset "
                    f"(got {type(dataset).__name__})."
                )
            metadata = dict(getattr(entry, "metadata", {}) or {})
        except Exception:
            if hasattr(self, "_last_fit_dataset") and isinstance(self._last_fit_dataset, xr.Dataset):
                dataset = self._last_fit_dataset.copy(deep=True)
                entry_id = "in_memory_last_fit"
            else:
                raise

        if "fit_uuid" not in dataset.attrs:
            for key_path in ("fit_uuid", "attrs.fit_uuid", "attr.fit_uuid"):
                value = self._get_nested_metadata_value(metadata, key_path)
                if value is not None:
                    dataset.attrs["fit_uuid"] = value
                    break
        if "task_name" not in dataset.attrs:
            dataset.attrs["task_name"] = fit_task_name
        dataset.attrs["tiled_entry_id"] = entry_id
        return dataset

    def _writedata(self, data):
        filename = pathlib.Path(self.config["filename"])
        filepath = pathlib.Path(self.config["filepath"])
        print(f"writing data to {filepath/filename}")
        with h5py.File(filepath / filename, "w") as f:
            f.create_dataset(str(uuid.uuid1()), data=data)

_DEFAULT_PORT = 5058
if __name__ == '__main__':
    from AFL.automation.shared.launcher import *
