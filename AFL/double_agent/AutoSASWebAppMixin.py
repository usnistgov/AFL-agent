import copy
import json
import pathlib
from importlib.resources import files
from typing import Any, Optional

import numpy as np
import sasmodels.core
import sasmodels.data
import sasmodels.direct_model
import xarray as xr
from jinja2 import Template
from AFL.double_agent.AutoSAS import SASFitter

try:
    from AFL.automation.APIServer.Driver import Driver  # type: ignore
except ModuleNotFoundError as exc:
    # Allow test/import in environments where AFL-automation is unavailable.
    if exc.name and exc.name.startswith("AFL.automation"):
        class Driver:  # type: ignore[override]
            @staticmethod
            def unqueued(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator
    else:
        raise


class AutoSASWebAppMixin:
    """Web-app endpoints and helper logic for AutoSAS model building and fitting."""

    static_dirs = {
        "autosas_webapp_js": pathlib.Path(__file__).parent / "driver_templates" / "autosas_webapp" / "js",
        "autosas_webapp_css": pathlib.Path(__file__).parent / "driver_templates" / "autosas_webapp" / "css",
        "autosas_webapp_vendor": pathlib.Path(__file__).parent / "driver_templates" / "autosas_webapp" / "vendor",
    }

    _DEFAULT_Q_MIN = 1e-3
    _DEFAULT_Q_MAX = 1.0

    def _autosas_webapp_init(self) -> None:
        """Initialize AutoSAS web-app state on a concrete driver instance."""
        self._last_fit_uuid: Optional[str] = None
        self._last_fit_summary: Optional[dict[str, Any]] = None
        self._autosas_input_dataset = None

        if getattr(self, "useful_links", None) is None:
            self.useful_links = {"AutoSAS Web App": "/autosas_webapp"}
        else:
            self.useful_links["AutoSAS Web App"] = "/autosas_webapp"

        if "autosas_tiled_entry_ids" not in self.config:
            self.config["autosas_tiled_entry_ids"] = []
        if "autosas_tiled_concat_dim" not in self.config:
            self.config["autosas_tiled_concat_dim"] = "sample"
        if "autosas_tiled_variable_prefix" not in self.config:
            self.config["autosas_tiled_variable_prefix"] = ""
        if "autosas_q_variable" not in self.config:
            self.config["autosas_q_variable"] = "q"
        if "autosas_sas_variable" not in self.config:
            self.config["autosas_sas_variable"] = "I"
        if "autosas_sas_err_variable" not in self.config:
            self.config["autosas_sas_err_variable"] = "dI"
        if "autosas_sas_resolution_variable" not in self.config:
            self.config["autosas_sas_resolution_variable"] = None

    @staticmethod
    def _coerce_json(value: Any, field_name: str) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            return json.loads(value)
        raise ValueError(f"{field_name} must be a dict/list or JSON string.")

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _fit_result_as_attrs_dataset(result: dict[str, Any]) -> xr.Dataset:
        """Encode fit result payload in Dataset.attrs for DataTiled-friendly ingestion."""
        dataset = xr.Dataset()
        attrs: dict[str, Any] = {}
        for key, value in result.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                attrs[key] = value
            else:
                attrs[key] = json.dumps(value)
        dataset.attrs.update(attrs)
        return dataset

    @staticmethod
    def _default_bounds(value: float) -> tuple[float, float]:
        if value > 0:
            return (value / 10.0, value * 10.0)
        if value < 0:
            return (value * 10.0, value / 10.0)
        return (-1.0, 1.0)

    @staticmethod
    def _model_template(model_name: str, index: Optional[int] = None) -> dict[str, Any]:
        kernel = sasmodels.core.load_model(model_name)
        pd_params = (
            kernel.info.parameters.pd_1d if hasattr(kernel.info.parameters, "pd_1d") else []
        )

        params: dict[str, Any] = {}
        for pname, pval in kernel.info.parameters.defaults.items():
            value = float(pval) if isinstance(pval, (int, float, np.floating, np.integer)) else pval
            params[pname] = {
                "value": value,
                "bounds": AutoSASWebAppMixin._default_bounds(float(value))
                if isinstance(value, (int, float, np.floating, np.integer))
                else None,
                "autosas": False,
                "use_bounds": False,
            }

            if pname in pd_params:
                params[f"{pname}_pd"] = {
                    "value": 0.0,
                    "bounds": (0.0, 1.0),
                    "autosas": False,
                    "use_bounds": False,
                }
                params[f"{pname}_pd_type"] = {
                    "value": "gaussian",
                    "options": ["gaussian", "rectangular", "schulz"],
                    "autosas": False,
                    "use_bounds": False,
                }
                params[f"{pname}_pd_n"] = {
                    "value": 35,
                    "bounds": (3, 200),
                    "autosas": False,
                    "use_bounds": False,
                }
                params[f"{pname}_pd_nsigma"] = {
                    "value": 3.0,
                    "bounds": (1.0, 10.0),
                    "autosas": False,
                    "use_bounds": False,
                }

        display_index = int(index) if index is not None else 1
        return {
            "name": f"{model_name}_{display_index}",
            "sasmodel": model_name,
            "q_min": AutoSASWebAppMixin._DEFAULT_Q_MIN,
            "q_max": AutoSASWebAppMixin._DEFAULT_Q_MAX,
            "params": params,
        }

    @staticmethod
    def _to_float(value: Any, field_name: str) -> float:
        try:
            return float(value)
        except Exception as exc:
            raise ValueError(f"{field_name} must be numeric.") from exc

    @staticmethod
    def _normalize_model_inputs(model_inputs: Any) -> list[dict[str, Any]]:
        parsed = AutoSASWebAppMixin._coerce_json(model_inputs, "model_inputs")
        if not isinstance(parsed, list):
            raise ValueError("model_inputs must be a list.")

        normalized: list[dict[str, Any]] = []
        for idx, model in enumerate(parsed):
            if not isinstance(model, dict):
                raise ValueError(f"model_inputs[{idx}] must be a dict.")

            name = str(model.get("name", "")).strip()
            sasmodel = str(model.get("sasmodel", "")).strip()
            if not name:
                raise ValueError(f"model_inputs[{idx}].name is required.")
            if not sasmodel:
                raise ValueError(f"model_inputs[{idx}].sasmodel is required.")

            try:
                sasmodels.core.load_model(sasmodel)
            except Exception as exc:
                raise ValueError(f"model_inputs[{idx}].sasmodel '{sasmodel}' is invalid.") from exc

            q_min = AutoSASWebAppMixin._to_float(model.get("q_min", AutoSASWebAppMixin._DEFAULT_Q_MIN), f"model_inputs[{idx}].q_min")
            q_max = AutoSASWebAppMixin._to_float(model.get("q_max", AutoSASWebAppMixin._DEFAULT_Q_MAX), f"model_inputs[{idx}].q_max")
            if q_min >= q_max:
                raise ValueError(f"model_inputs[{idx}] must satisfy q_min < q_max.")

            fit_params = model.get("fit_params", {})
            if not isinstance(fit_params, dict):
                raise ValueError(f"model_inputs[{idx}].fit_params must be a dict.")

            normalized_fit_params: dict[str, dict[str, Any]] = {}
            for pname, pinfo in fit_params.items():
                if not isinstance(pinfo, dict):
                    raise ValueError(f"model_inputs[{idx}].fit_params['{pname}'] must be a dict.")
                if "value" not in pinfo:
                    raise ValueError(f"model_inputs[{idx}].fit_params['{pname}'] missing value.")

                value = AutoSASWebAppMixin._to_float(
                    pinfo["value"], f"model_inputs[{idx}].fit_params['{pname}'].value"
                )
                normalized_param: dict[str, Any] = {"value": value}

                bounds = pinfo.get("bounds")
                if bounds is not None:
                    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                        raise ValueError(
                            f"model_inputs[{idx}].fit_params['{pname}'].bounds must be [min,max]."
                        )
                    b0 = AutoSASWebAppMixin._to_float(bounds[0], f"bounds[0] for {pname}")
                    b1 = AutoSASWebAppMixin._to_float(bounds[1], f"bounds[1] for {pname}")
                    if b0 >= b1:
                        raise ValueError(
                            f"model_inputs[{idx}].fit_params['{pname}'].bounds must satisfy min < max."
                        )
                    normalized_param["bounds"] = (b0, b1)

                if "pd" in pinfo:
                    normalized_param["pd"] = AutoSASWebAppMixin._to_float(
                        pinfo["pd"], f"model_inputs[{idx}].fit_params['{pname}'].pd"
                    )
                if "pd_type" in pinfo:
                    normalized_param["pd_type"] = str(pinfo["pd_type"])
                if "pd_n" in pinfo:
                    normalized_param["pd_n"] = int(pinfo["pd_n"])
                if "pd_nsigma" in pinfo:
                    normalized_param["pd_nsigma"] = AutoSASWebAppMixin._to_float(
                        pinfo["pd_nsigma"], f"model_inputs[{idx}].fit_params['{pname}'].pd_nsigma"
                    )

                normalized_fit_params[str(pname)] = normalized_param

            normalized.append(
                {
                    "name": name,
                    "sasmodel": sasmodel,
                    "q_min": q_min,
                    "q_max": q_max,
                    "fit_params": normalized_fit_params,
                }
            )

        return normalized

    @staticmethod
    def _summary_from_fitter(fitter: Any) -> dict[str, Any]:
        report = getattr(fitter, "report", {}) or {}
        best = report.get("best_fits", {}) if isinstance(report, dict) else {}
        probabilities = report.get("probabilities", [])
        all_chisq = report.get("all_chisq", [])
        if isinstance(probabilities, np.ndarray):
            probabilities = probabilities.tolist()
        if isinstance(all_chisq, np.ndarray):
            all_chisq = all_chisq.tolist()

        return {
            "status": "success",
            "best_model_names": best.get("model_name", []),
            "best_chisq": best.get("lowest_chisq", []),
            "probabilities": probabilities,
            "all_chisq": all_chisq,
        }

    @staticmethod
    def _parse_entry_ids(entry_ids: Any) -> list[str]:
        if entry_ids is None:
            return []
        if isinstance(entry_ids, list):
            return [str(v).strip() for v in entry_ids if str(v).strip()]
        if isinstance(entry_ids, str):
            return [part.strip() for part in entry_ids.replace(",", "\n").splitlines() if part.strip()]
        raise ValueError("entry_ids must be a list or a newline/comma-delimited string.")

    @Driver.unqueued(render_hint="html")
    def autosas_webapp(self, **kwargs):
        """Serve AutoSAS web application HTML."""
        template_path = (
            files("AFL.double_agent.driver_templates")
            .joinpath("autosas_webapp")
            .joinpath("autosas_webapp.html")
        )
        template = Template(template_path.read_text())
        return template.render()

    @Driver.unqueued()
    def autosas_get_bootstrap(self, **kwargs):
        """Return bootstrap state for the AutoSAS web app."""
        try:
            available_models = sorted(sasmodels.core.list_models())
        except Exception:
            available_models = [
                "sphere",
                "cylinder",
                "ellipsoid",
                "core_shell_sphere",
                "core_shell_cylinder",
                "power_law",
            ]

        preview_data = None
        fitter = getattr(self, "fitter", None)
        loaded_dataset_summary = None
        if self._autosas_input_dataset is not None:
            ds = self._autosas_input_dataset
            html_repr = ds._repr_html_() if hasattr(ds, "_repr_html_") else f"<pre>{str(ds)}</pre>"
            loaded_dataset_summary = {
                "dims": dict(ds.sizes),
                "data_vars": list(ds.data_vars),
                "coords": list(ds.coords),
                "html": html_repr,
            }
        if fitter is not None and getattr(fitter, "sasdata", None):
            try:
                first_data = fitter.sasdata[0]
                preview_data = {
                    "q": np.asarray(first_data.x).tolist(),
                    "I": np.asarray(first_data.y).tolist(),
                }
            except Exception:
                preview_data = None

        return {
            "status": "success",
            "available_models": available_models,
            "model_inputs": copy.deepcopy(self.config.get("model_inputs", [])),
            "fit_method": copy.deepcopy(self.config.get("fit_method", {})),
            "defaults": {
                "q_min": self.config.get("q_min", self._DEFAULT_Q_MIN),
                "q_max": self.config.get("q_max", self._DEFAULT_Q_MAX),
            },
            "has_fitter": fitter is not None,
            "data_preview": preview_data,
            "loaded_sample_total": len(getattr(fitter, "sasdata", [])) if fitter is not None else 0,
            "loaded_dataset": loaded_dataset_summary,
            "last_fit_uuid": self._last_fit_uuid,
            "last_fit_summary": copy.deepcopy(self._last_fit_summary),
            "tiled_context": {
                "entry_ids": copy.deepcopy(self.config.get("autosas_tiled_entry_ids", [])),
                "concat_dim": self.config.get("autosas_tiled_concat_dim", "sample"),
                "variable_prefix": self.config.get("autosas_tiled_variable_prefix", ""),
                "q_variable": self.config.get("autosas_q_variable", "q"),
                "sas_variable": self.config.get("autosas_sas_variable", "I"),
                "sas_err_variable": self.config.get("autosas_sas_err_variable", "dI"),
                "sas_resolution_variable": self.config.get("autosas_sas_resolution_variable", None),
            },
        }

    @Driver.unqueued()
    def autosas_get_loaded_sample(self, sample_index: int = 0, **kwargs):
        """Return one loaded SAS sample for UI navigation."""
        fitter = getattr(self, "fitter", None)
        if fitter is None or not getattr(fitter, "sasdata", None):
            return {"status": "error", "message": "No loaded SAS data context."}

        total = len(fitter.sasdata)
        idx = int(sample_index)
        if idx < 0:
            idx = 0
        if idx >= total:
            idx = total - 1

        sample = fitter.sasdata[idx]
        return {
            "status": "success",
            "sample_index": idx,
            "sample_total": total,
            "q": np.asarray(sample.x).tolist(),
            "I": np.asarray(sample.y).tolist(),
        }

    @Driver.unqueued()
    def autosas_get_model_template(self, sasmodel: str = "sphere", index: Optional[int] = None, **kwargs):
        """Return default editable parameter template for a sasmodel."""
        try:
            model_template = self._model_template(sasmodel, index=index)
            return {"status": "success", "model": model_template}
        except Exception as exc:
            return {"status": "error", "message": f"Unable to build model template: {exc}"}

    @Driver.unqueued()
    def autosas_preview_model(
        self,
        model_config: Any = None,
        q_points: Any = 300,
        q_global_min: Any = None,
        q_global_max: Any = None,
        sample_index: Any = 0,
        **kwargs,
    ):
        """Return model q/I preview for a model configuration."""
        try:
            model = self._coerce_json(model_config, "model_config")
            if not isinstance(model, dict):
                raise ValueError("model_config must be a dict.")

            sasmodel = str(model.get("sasmodel", "")).strip()
            if not sasmodel:
                raise ValueError("model_config.sasmodel is required.")

            kernel = sasmodels.core.load_model(sasmodel)

            qmin = self._to_float(model.get("q_min", q_global_min or self._DEFAULT_Q_MIN), "q_min")
            qmax = self._to_float(model.get("q_max", q_global_max or self._DEFAULT_Q_MAX), "q_max")
            if qmin <= 0 or qmax <= 0:
                raise ValueError("q_min and q_max must be > 0 for log-space preview.")
            if qmin >= qmax:
                raise ValueError("q_min must be < q_max.")

            qn = max(30, int(q_points))
            q = np.logspace(np.log10(qmin), np.log10(qmax), qn)

            params_input = model.get("params", {})
            if not isinstance(params_input, dict):
                raise ValueError("model_config.params must be a dict.")

            params = {}
            for key, value in params_input.items():
                if isinstance(value, dict):
                    params[str(key)] = value.get("value")
                else:
                    params[str(key)] = value

            data = sasmodels.data.empty_data1D(q)
            calculator = sasmodels.direct_model.DirectModel(data, kernel)
            intensity = calculator(**params)

            chi_q = []
            chi_vals = []
            fitter = getattr(self, "fitter", None)
            if fitter is not None and getattr(fitter, "sasdata", None):
                total = len(fitter.sasdata)
                idx = int(sample_index)
                if idx < 0:
                    idx = 0
                if idx >= total:
                    idx = total - 1

                sample = fitter.sasdata[idx]
                x_data = np.asarray(sample.x)
                y_data = np.asarray(sample.y)
                fit_mask = (x_data >= qmin) & (x_data <= qmax)
                q_fit = x_data[fit_mask]
                y_fit = y_data[fit_mask]

                if q_fit.size > 0:
                    fit_data = sasmodels.data.empty_data1D(q_fit)
                    fit_calc = sasmodels.direct_model.DirectModel(fit_data, kernel)
                    y_model = np.asarray(fit_calc(**params))

                    dy_full = getattr(sample, "dy", None)
                    if dy_full is not None:
                        dy_fit = np.asarray(dy_full)[fit_mask]
                        valid = np.isfinite(dy_fit) & (dy_fit > 0)
                        chi = np.full(y_model.shape, np.nan, dtype=float)
                        chi[valid] = (y_fit[valid] - y_model[valid]) / dy_fit[valid]
                    else:
                        chi = y_fit - y_model

                    chi_q = np.asarray(q_fit).tolist()
                    chi_vals = np.asarray(chi).tolist()

            return {
                "status": "success",
                "q": np.asarray(q).tolist(),
                "intensity": np.asarray(intensity).tolist(),
                "chi_q": chi_q,
                "chi": chi_vals,
                "chisq_q": chi_q,  # backward-compatible alias
                "chisq": chi_vals,  # backward-compatible alias
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Preview failed: {exc}",
                "q": [],
                "intensity": [],
                "chi_q": [],
                "chi": [],
                "chisq_q": [],
                "chisq": [],
            }

    @Driver.unqueued()
    def autosas_validate_model_inputs(self, model_inputs: Any = None, **kwargs):
        """Validate model_inputs payload shape and values."""
        try:
            normalized = self._normalize_model_inputs(model_inputs)
            return {
                "status": "success",
                "message": f"Validated {len(normalized)} model(s).",
                "model_inputs": normalized,
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def autosas_apply_model_inputs(self, model_inputs: Any = None, **kwargs):
        """Persist model_inputs into driver config."""
        normalized = self._normalize_model_inputs(model_inputs)
        self.config["model_inputs"] = normalized
        return {
            "status": "success",
            "message": f"Saved {len(normalized)} model(s) to config.model_inputs.",
            "model_inputs": normalized,
        }

    def autosas_set_data_context(
        self,
        db_uuid: str,
        sample_dim: str = "sample",
        q_variable: str = "q",
        sas_variable: str = "I",
        sas_err_variable: str = "dI",
        sas_resolution_variable: Optional[str] = None,
        **kwargs,
    ):
        """Set fitter SAS data context from a dataset UUID."""
        if not db_uuid:
            raise ValueError("db_uuid is required.")

        self.set_sasdata(
            db_uuid=db_uuid,
            sample_dim=sample_dim,
            q_variable=q_variable,
            sas_variable=sas_variable,
            sas_err_variable=sas_err_variable,
            sas_resolution_variable=sas_resolution_variable,
        )

        n_samples = len(getattr(self.fitter, "sasdata", [])) if self.fitter is not None else 0
        return {
            "status": "success",
            "message": f"SAS data loaded for fitting ({n_samples} sample(s)).",
            "n_samples": n_samples,
        }

    def autosas_set_tiled_data_context(
        self,
        entry_ids: Any = None,
        concat_dim: str = "sample",
        variable_prefix: str = "",
        q_variable: str = "q",
        sas_variable: str = "I",
        sas_err_variable: str = "dI",
        sas_resolution_variable: Optional[str] = None,
        **kwargs,
    ):
        """Assemble dataset from Tiled entry IDs and set fitter SAS data context."""
        parsed_entry_ids = self._parse_entry_ids(entry_ids)
        if not parsed_entry_ids:
            raise ValueError("entry_ids is required and must contain at least one entry id.")

        if not concat_dim:
            raise ValueError("concat_dim is required.")
        if not q_variable:
            raise ValueError("q_variable is required.")
        if not sas_variable:
            raise ValueError("sas_variable is required.")
        if not sas_err_variable:
            raise ValueError("sas_err_variable is required.")

        dataset = self.tiled_concat_datasets(
            entry_ids=parsed_entry_ids,
            concat_dim=concat_dim,
            variable_prefix=variable_prefix or "",
        )
        dataset = dataset.reset_coords()

        self._autosas_input_dataset = dataset
        self.fitter = SASFitter(
            model_inputs=self.config["model_inputs"],
            fit_method=self.config["fit_method"],
            q_min=self.config["q_min"],
            q_max=self.config["q_max"],
            resolution=self.config["resolution"],
        )
        self.fitter.set_sasdata(
            dataset=dataset,
            sample_dim=concat_dim,
            q_variable=q_variable,
            sas_variable=sas_variable,
            sas_err_variable=sas_err_variable,
            sas_resolution_variable=(sas_resolution_variable or None),
        )

        self.config["autosas_tiled_entry_ids"] = parsed_entry_ids
        self.config["autosas_tiled_concat_dim"] = concat_dim
        self.config["autosas_tiled_variable_prefix"] = variable_prefix or ""
        self.config["autosas_q_variable"] = q_variable
        self.config["autosas_sas_variable"] = sas_variable
        self.config["autosas_sas_err_variable"] = sas_err_variable
        self.config["autosas_sas_resolution_variable"] = sas_resolution_variable or None

        html_repr = dataset._repr_html_() if hasattr(dataset, "_repr_html_") else f"<pre>{str(dataset)}</pre>"
        return {
            "status": "success",
            "message": f"Assembled {len(parsed_entry_ids)} tiled dataset(s) and loaded fitter context.",
            "n_samples": len(getattr(self.fitter, "sasdata", [])),
            "dims": dict(dataset.sizes),
            "data_vars": list(dataset.data_vars),
            "coords": list(dataset.coords),
            "html": html_repr,
        }

    def autosas_clear_data_context(self, **kwargs):
        """Clear loaded SAS data context used by the AutoSAS web app."""
        self.fitter = None
        self._autosas_input_dataset = None
        self.config["autosas_tiled_entry_ids"] = []

        return {
            "status": "success",
            "message": "Cleared loaded SAS data context.",
            "n_samples": 0,
        }

    def autosas_run_fit(
        self,
        parallel: Any = False,
        fit_method: Any = None,
        model_inputs: Any = None,
        sample_index: Any = None,
        return_dataset: Any = False,
        **kwargs,
    ):
        """Run fitting for requested model_inputs and sample, or full configured fit when omitted."""
        parallel_flag = self._coerce_bool(parallel)
        return_dataset_flag = self._coerce_bool(return_dataset)

        fit_method_payload = fit_method
        if isinstance(fit_method, str) and fit_method.strip():
            fit_method_payload = json.loads(fit_method)
        if self.fitter is None or not getattr(self.fitter, "sasdata", None):
            raise ValueError("No SAS data set. Use set_sasdata first.")

        if model_inputs is None and sample_index is None:
            fit_result = self.fit_models(parallel=parallel_flag, fit_method=fit_method_payload)
            fit_uuid = None
            if isinstance(fit_result, xr.Dataset):
                fit_uuid = fit_result.attrs.get("fit_uuid")
            elif isinstance(fit_result, str):
                fit_uuid = fit_result
            summary = None
            if self.fitter is not None:
                summary = self._summary_from_fitter(self.fitter)
                summary["fit_uuid"] = fit_uuid
            self._last_fit_uuid = fit_uuid
            self._last_fit_summary = summary
            result = {
                "status": "success",
                "fit_uuid": fit_uuid,
                "summary": summary,
            }
            if return_dataset_flag:
                return self._fit_result_as_attrs_dataset(result)
            return result

        if model_inputs is None:
            normalized_inputs = copy.deepcopy(self.config.get("model_inputs", []))
        else:
            normalized_inputs = self._normalize_model_inputs(model_inputs)
        if not normalized_inputs:
            raise ValueError("At least one model input is required.")

        idx = 0 if sample_index is None else int(sample_index)
        total_samples = len(self.fitter.sasdata)
        if idx < 0:
            idx = 0
        if idx >= total_samples:
            idx = total_samples - 1

        fit_target = copy.deepcopy(self.fitter.sasdata[idx])
        fit_runner = SASFitter(
            model_inputs=normalized_inputs,
            fit_method=fit_method_payload or self.config["fit_method"],
            q_min=self.config["q_min"],
            q_max=self.config["q_max"],
            resolution=self.config["resolution"],
        )
        fit_runner.sasdata = [fit_target]

        fit_uuid, _ = fit_runner.fit_models(parallel=parallel_flag, fit_method=fit_method_payload)

        summary = self._summary_from_fitter(fit_runner)
        summary["fit_uuid"] = fit_uuid

        fitted_params = None
        model_curve_q = []
        model_curve_i = []
        chi_curve_q = []
        chi_curve_vals = []
        if fit_runner.fit_results and fit_runner.fit_results[0]:
            fitted_params = copy.deepcopy(fit_runner.fit_results[0][0].get("output_fit_params", {}))
        if fit_runner.fitted_models and fit_runner.fitted_models[0]:
            fitted_model = fit_runner.fitted_models[0][0]
            model_curve_q = np.asarray(getattr(fitted_model, "model_q", [])).tolist()
            model_curve_i = np.asarray(getattr(fitted_model, "model_I", [])).tolist()

            data_obj = getattr(fitted_model, "data", None)
            if data_obj is not None:
                data_x = np.asarray(getattr(data_obj, "x", []))
                data_y = np.asarray(getattr(data_obj, "y", []))
                mask = np.asarray(getattr(data_obj, "mask", np.zeros_like(data_x, dtype=bool)))
                if mask.shape == data_x.shape:
                    keep = mask == 0
                else:
                    keep = np.ones_like(data_x, dtype=bool)
                y_fit = data_y[keep]
                model_y = np.asarray(getattr(fitted_model, "model_I", []))

                if y_fit.shape == model_y.shape and y_fit.size > 0:
                    dy_full = getattr(data_obj, "dy", None)
                    if dy_full is not None:
                        dy_fit = np.asarray(dy_full)[keep]
                        valid = np.isfinite(dy_fit) & (dy_fit > 0)
                        chi_arr = np.full(model_y.shape, np.nan, dtype=float)
                        chi_arr[valid] = (y_fit[valid] - model_y[valid]) / dy_fit[valid]
                    else:
                        chi_arr = y_fit - model_y

                    chi_curve_q = model_curve_q
                    chi_curve_vals = np.asarray(chi_arr).tolist()

        self._last_fit_uuid = fit_uuid
        self._last_fit_summary = summary

        result = {
            "status": "success",
            "fit_uuid": fit_uuid,
            "summary": summary,
            "sample_index": idx,
            "fitted_model_name": normalized_inputs[0].get("name"),
            "fitted_params": fitted_params,
            "model_curve": {
                "q": model_curve_q,
                "intensity": model_curve_i,
            },
            "chi_curve": {
                "q": chi_curve_q,
                "chi": chi_curve_vals,
            },
            "chisq_curve": {  # backward-compatible alias
                "q": chi_curve_q,
                "chisq": chi_curve_vals,
            },
        }
        if return_dataset_flag:
            return self._fit_result_as_attrs_dataset(result)
        return result

    @Driver.unqueued()
    def autosas_run_fit_unqueued(
        self,
        parallel: Any = False,
        fit_method: Any = None,
        model_inputs: Any = None,
        sample_index: Any = None,
        return_dataset: Any = True,
        **kwargs,
    ):
        """Run AutoSAS fit synchronously via unqueued route.

        Defaults to returning an empty xr.Dataset with fit payload encoded in .attrs,
        which is useful for DataTiled ingestion.
        """
        return self.autosas_run_fit(
            parallel=parallel,
            fit_method=fit_method,
            model_inputs=model_inputs,
            sample_index=sample_index,
            return_dataset=return_dataset,
            **kwargs,
        )

    @Driver.unqueued()
    def autosas_last_fit_summary(self, **kwargs):
        """Return the most recently captured fit summary."""
        if self._last_fit_summary is not None:
            return {
                "status": "success",
                "fit_uuid": self._last_fit_uuid,
                "summary": copy.deepcopy(self._last_fit_summary),
            }

        fitter = getattr(self, "fitter", None)
        if fitter is not None and getattr(fitter, "report", None):
            summary = self._summary_from_fitter(fitter)
            return {
                "status": "success",
                "fit_uuid": self._last_fit_uuid,
                "summary": summary,
            }

        return {
            "status": "error",
            "message": "No fit summary available yet.",
        }

    @Driver.unqueued()
    def autosas_get_fit_entry_id(
        self,
        fit_uuid: Optional[str] = None,
        fit_task_name: str = "fit_models",
        **kwargs,
    ):
        """Resolve Tiled entry_id for a fit UUID by searching metadata attrs.fit_uuid."""
        target_fit_uuid = fit_uuid or getattr(self, "_last_fit_uuid", None)
        if not target_fit_uuid:
            return {"status": "error", "message": "fit_uuid is required and no last fit UUID is available."}

        if not hasattr(self, "_iter_fit_entries") or not hasattr(self, "_select_latest_entry"):
            return {"status": "error", "message": "Driver does not implement Tiled fit entry lookup helpers."}

        try:
            entries = self._iter_fit_entries(
                fit_uuid=target_fit_uuid,
                task_name=fit_task_name,
                allow_task_fallback=False,
            )
            entry_id, _ = self._select_latest_entry(entries)
            return {
                "status": "success",
                "fit_uuid": target_fit_uuid,
                "entry_id": entry_id,
            }
        except Exception as exc:
            return {
                "status": "error",
                "fit_uuid": target_fit_uuid,
                "message": f"Could not resolve Tiled entry for fit UUID '{target_fit_uuid}': {exc}",
            }
