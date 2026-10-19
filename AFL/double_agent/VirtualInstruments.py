"""Virtual instruments used in AFL-agent examples and local simulations."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from AFL.double_agent._automation_compat import Driver

try:
    from AFL.automation.instrument.VirtualInstrument import VirtualInstrument
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("AFL.automation"):
        # The virtual instrument API is useful without an APIServer. Preserve
        # its public inheritance shape while using the same lightweight Driver
        # fallback as AgentDriver when AFL-automation is not installed.
        VirtualInstrument = Driver
    else:
        raise


def _driver_init(instrument, name: str, overrides: dict | None, afl_home: str | Path | None) -> None:
    """Call ``Driver.__init__`` passing ``afl_home`` only when this Driver accepts it."""
    kwargs = {}
    params = inspect.signature(Driver.__init__).parameters
    if afl_home is not None and (
        "afl_home" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    ):
        kwargs["afl_home"] = afl_home
    Driver.__init__(instrument, name=name, defaults=instrument.gather_defaults(), overrides=overrides, **kwargs)


class GaussianVirtualInstrument(VirtualInstrument):
    """Generate one-dimensional Gaussian spectra from ``mu`` and ``sigma``.

    ``measure`` accepts one parameter row or a batch with columns ``mu`` and
    ``sigma``. It returns an xarray dataset ready for AFL-agent pipelines.
    """

    defaults = {
        "wavelength_min": 0.0,
        "wavelength_max": 1.0,
        "n_wavelengths": 201,
        "amplitude": 1.0,
        "noise": 0.0,
        "random_seed": 7,
    }

    def __init__(
        self,
        overrides: dict | None = None,
        afl_home: str | Path | None = None,
    ) -> None:
        # VirtualInstrument's ScatteringInstrument initializer requires the
        # optional pyFAI package. This spectral instrument needs only Driver.
        self.app = None
        _driver_init(self, "GaussianVirtualInstrument", overrides, afl_home)
        self.__instrument_name__ = "Gaussian Virtual Spectrometer"
        self.status_txt = "Idle"
        self._rng = np.random.default_rng(int(self.config["random_seed"]))

    @property
    def wavelength(self) -> np.ndarray:
        """Return the fixed coordinate used by every synthetic measurement."""

        return np.linspace(
            float(self.config["wavelength_min"]),
            float(self.config["wavelength_max"]),
            int(self.config["n_wavelengths"]),
        )

    def measure(self, parameters: np.ndarray) -> xr.Dataset:
        """Measure one or more ``(mu, sigma)`` rows."""

        parameters = np.atleast_2d(np.asarray(parameters, dtype=float))
        if parameters.ndim != 2 or parameters.shape[1] != 2:
            raise ValueError("parameters must have shape (n_samples, 2)")
        if not np.isfinite(parameters).all():
            raise ValueError("mu and sigma must be finite")
        if np.any(parameters[:, 1] <= 0.0):
            raise ValueError("sigma must be greater than zero")

        mu = parameters[:, 0, np.newaxis]
        sigma = parameters[:, 1, np.newaxis]
        spectrum = float(self.config["amplitude"]) * np.exp(
            -0.5 * ((self.wavelength[np.newaxis, :] - mu) / sigma) ** 2
        )
        noise = float(self.config["noise"])
        if noise > 0.0:
            spectrum = spectrum + self._rng.normal(scale=noise, size=spectrum.shape)

        return xr.Dataset(
            data_vars={
                "parameters": (("sample", "component"), parameters),
                "spectrum": (("sample", "wavelength"), spectrum),
            },
            coords={
                "sample": np.arange(parameters.shape[0]),
                "component": ["mu", "sigma"],
                "wavelength": self.wavelength,
            },
        )

    def status(self) -> list[str]:
        return [self.status_txt]


class VirtualSASInstrument(VirtualInstrument):
    """Base class for simulating small-angle scattering from a labelled phase diagram.

    A port of the ``VirtualSAS_theory`` simulator used in AFL-bioformulations
    (``bioformulationagents.afl.helpers.get_simulator``). Phase regions are traced as
    alpha-shape hulls from a labelled boundary dataset. A requested composition is
    located in those hulls and a SasView model assigned to that phase generates the
    scattering curve on the q-grid of the loaded reference instrument configurations.

    This class carries only the system-independent configuration (hull tracing,
    phase lookup tolerance, noise, random seed). It is not tied to any particular
    dataset: either subclass it and add ``boundary_dataset``, ``component_limits``,
    ``reference_configurations`` and ``sasview_models`` to ``defaults`` (see
    :class:`ProteinGlycerolTemperaturePhaseMap`), or instantiate it directly and call
    :meth:`load_boundary_dataset`, :meth:`trace_boundaries`,
    :meth:`add_configuration_from_file` / :meth:`add_configuration` and
    :meth:`add_sasview_model` yourself.

    Requires the optional ``alphashape`` and ``trimesh`` packages
    (``pip install AFL-agent[virtual-sas]``) and ``sasmodels``.

    Parameters
    ----------
    overrides : dict, optional
        Override any key in ``defaults``.
    afl_home : str | Path, optional
        Passed to the AFL-automation Driver when available.

    Notes
    -----
    Phase lookup mirrors the original: with ``fast_locate`` the first hull containing the
    point (or within ``locate_tolerance`` of it, in min-max scaled space) wins, so points
    close to a boundary may be attributed to a neighbouring phase. Compositions outside
    every hull are assigned the nearest phase and flagged with ``in_hull=False``; the
    original raised an error in that case.

    ``measure`` takes composition rows ordered like the ``component`` coordinate of the
    boundary dataset and returns a dataset with ``I``, ``I_noiseless``, ``dI`` over
    ``(sample, q)`` and integer ``labels``. ``simple_expose`` keeps the single-composition
    dict interface of the original simulator.

    Optional configuration keys consumed by the constructor when present
    (subclasses provide them through ``defaults``):

    ``boundary_dataset``
        ``xarray.Dataset``, NetCDF path, or stem of a file in ``AFL/double_agent/data``.
    ``component_limits``
        Mapping of component name to ``(min, max)`` used to crop the dataset before
        tracing hulls.
    ``reference_configurations``
        List of ``.ABS`` files (paths or names relative to ``AFL/double_agent/data``).
    ``sasview_models``
        Mapping of phase label to ``{"model_name": ..., "model_kw": {...}}``.
    """

    instrument_name = "Virtual SAS Instrument"

    defaults = {
        "hull_tracing_ratio": 0.25,
        "locate_tolerance": 0.1,
        "fast_locate": True,
        "noise": 0.0,
        "random_seed": 7,
    }

    def __init__(
        self,
        overrides: dict | None = None,
        afl_home: str | Path | None = None,
    ) -> None:
        self.app = None
        _driver_init(self, type(self).__name__, overrides, afl_home)
        self.__instrument_name__ = self.instrument_name
        self.status_txt = "Idle"
        self._rng = np.random.default_rng(int(self.config["random_seed"]))

        self.hulls: dict[str, object] = {}
        self.reference_data: list = []
        self.sasmodels: dict[str, dict] = {}
        self.boundary_dataset: xr.Dataset | None = None
        self.scaler = None

        self.setup_from_config()

    def setup_from_config(self) -> None:
        """Load the dataset, trace hulls and register models described in ``self.config``.

        Each step is skipped when its configuration key is absent or empty, so the
        bare base class can be set up manually after construction.
        """
        boundary_dataset = self.config.get("boundary_dataset")
        if boundary_dataset is not None:
            self.load_boundary_dataset(boundary_dataset)
            self.trace_boundaries(hull_tracing_ratio=float(self.config["hull_tracing_ratio"]))
        for relative_path in self.config.get("reference_configurations") or []:
            self.add_configuration_from_file(relative_path, reset=False)
        for label, spec in (self.config.get("sasview_models") or {}).items():
            self.add_sasview_model(label=str(label), **spec)

    # ------------------------------------------------------------------ setup
    @property
    def components(self) -> list[str]:
        """Component names in the order ``measure`` expects."""
        ds = self.boundary_dataset
        return [str(c) for c in ds[ds.attrs.get("components_dim", "component")].values]

    def load_boundary_dataset(self, dataset: str | Path | xr.Dataset) -> None:
        """Load the labelled dataset used to trace phase hulls.

        ``dataset`` may be an ``xarray.Dataset``, a path to a NetCDF file, or the stem of
        a file in ``AFL/double_agent/data``.
        """
        if isinstance(dataset, xr.Dataset):
            ds = dataset.copy()
        else:
            path = Path(dataset)
            if not path.exists():
                from AFL.double_agent.data import get_data_dir

                path = get_data_dir() / f"{dataset}.nc"
            ds = xr.load_dataset(path)
        labels_var = ds.attrs.get("labels", "labels")
        ds["labels"] = ds[labels_var].astype(str)
        self.boundary_dataset = ds

    def trace_boundaries(
        self,
        hull_tracing_ratio: float = 0.1,
        drop_phases: list[str] | None = None,
        reset: bool = True,
    ) -> None:
        """Fit one alpha-shape hull per phase label in min-max scaled composition space."""
        try:
            import alphashape
            from sklearn.preprocessing import MinMaxScaler
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
            raise ModuleNotFoundError(
                f"{type(self).__name__} requires alphashape, trimesh and scikit-learn. "
                "Install them with `pip install AFL-agent[virtual-sas]`."
            ) from exc

        if self.boundary_dataset is None:
            raise ValueError("Call load_boundary_dataset before trace_boundaries")
        drop_phases = [str(p) for p in (drop_phases or [])]
        if reset:
            self.hulls = {}

        ds = self.boundary_dataset
        composition = ds[ds.attrs.get("components", "composition")]
        component_dim = ds.attrs.get("components_dim", "component")
        sample_dim = [d for d in composition.dims if d != component_dim][0]

        limits = self.config.get("component_limits") or {}
        mask = xr.ones_like(composition.isel({component_dim: 0}), dtype=bool)
        for component in composition[component_dim].values:
            if component in limits:
                lo, hi = limits[component]
                values = composition.sel({component_dim: component})
                mask = mask & (values >= lo) & (values <= hi)

        xyz = composition.transpose(sample_dim, component_dim).values[mask.values]
        labels = ds["labels"].values[mask.values]
        self.scaler = MinMaxScaler().fit(xyz)
        scaled = self.scaler.transform(xyz)
        # jitter by 0.01% of the range to avoid degenerate (collinear) alpha-shape facets
        scaled = scaled + self._rng.normal(0.0, np.ptp(scaled, axis=0) * 1e-4, scaled.shape)

        for label in np.unique(labels):
            if label in drop_phases:
                continue
            self.hulls[str(label)] = alphashape.alphashape(scaled[labels == label], hull_tracing_ratio)

    def add_configuration(self, q, I, dI, dq, reset: bool = True) -> None:
        """Add a reference instrument configuration that defines a q-grid and uncertainties."""
        import sasmodels.data

        if reset:
            self.reference_data = []
        self.reference_data.append(
            sasmodels.data.Data1D(x=np.asarray(q, float), y=np.asarray(I, float), dy=np.asarray(dI, float), dx=np.asarray(dq, float))
        )

    def add_configuration_from_file(self, path: str | Path, reset: bool = True) -> None:
        """Read a whitespace-delimited ``.ABS`` file with a ``q I dI dq`` header."""
        path = Path(path)
        if not path.exists():
            from AFL.double_agent.data import get_data_dir

            path = get_data_dir() / path
        table = np.genfromtxt(path, names=True)
        self.add_configuration(q=table["q"], I=table["I"], dI=table["dI"], dq=table["dq"], reset=reset)

    def add_sasview_model(self, label: str, model_name: str, model_kw: dict) -> None:
        """Assign a SasView model to a phase label. Must be called after reference data is loaded."""
        import sasmodels.core
        import sasmodels.direct_model

        kernel = sasmodels.core.build_model(sasmodels.core.load_model_info(model_name))
        calculators = [sasmodels.direct_model.DirectModel(sasdata, kernel) for sasdata in self.reference_data]
        self.sasmodels[str(label)] = {
            "name": model_name,
            "kw": dict(model_kw),
            "calculators": calculators,
            "sasdata": list(self.reference_data),
        }

    # --------------------------------------------------------------- physics
    def locate(self, composition) -> list[str]:
        """Return the phase labels whose hulls contain ``composition`` (one row)."""
        from trimesh.proximity import signed_distance

        if not self.hulls:
            raise ValueError("Call trace_boundaries before locate")
        point = self.scaler.transform(np.atleast_2d(np.asarray(composition, float)))
        tolerance = float(self.config["locate_tolerance"])
        phases = []
        for phase, hull in self.hulls.items():
            dist = float(signed_distance(hull, point)[0])
            if dist > 0 or abs(dist) < tolerance:
                phases.append(phase)
                if self.config["fast_locate"]:
                    break
        if len(phases) > 1:
            warnings.warn("Composition lies in multiple phases; hulls likely overlap", stacklevel=2)
        return phases

    def nearest_phase(self, composition) -> str:
        """Label of the hull closest to ``composition`` (used when no hull contains it)."""
        from trimesh.proximity import signed_distance

        point = self.scaler.transform(np.atleast_2d(np.asarray(composition, float)))
        distances = {phase: float(signed_distance(hull, point)[0]) for phase, hull in self.hulls.items()}
        return max(distances, key=distances.get)

    def label_for(self, composition) -> tuple[str, bool]:
        """Phase label for one composition row and whether it fell inside a traced hull."""
        phases = self.locate(composition)
        if phases:
            return phases[0], True
        return self.nearest_phase(composition), False

    def generate(self, label: str):
        """Scattering for one phase label: ``(I, I_noiseless, dI)`` arrays sorted by q, plus q."""
        if str(label) not in self.sasmodels:
            raise KeyError(f"No SasView model registered for phase label {label!r}")
        spec = self.sasmodels[str(label)]
        noise = float(self.config["noise"])

        q_parts, I_parts, I0_parts, dI_parts = [], [], [], []
        for sasdata, calc in zip(spec["sasdata"], spec["calculators"]):
            I_noiseless = calc(**spec["kw"])
            dI_model = sasdata.dy * np.sqrt(I_noiseless / sasdata.y)
            mean_var = np.mean(dI_model * dI_model / I_noiseless)
            dI = sasdata.dy * noise / mean_var
            I = self._rng.normal(loc=I_noiseless, scale=dI) if noise > 0 else I_noiseless.copy()
            q_parts.append(sasdata.x)
            I_parts.append(I)
            I0_parts.append(I_noiseless)
            dI_parts.append(dI)

        q = np.concatenate(q_parts)
        order = np.argsort(q, kind="stable")
        return (
            np.concatenate(I_parts)[order],
            np.concatenate(I0_parts)[order],
            np.concatenate(dI_parts)[order],
            q[order],
        )

    # ------------------------------------------------------------------- API
    def measure(self, parameters) -> xr.Dataset:
        """Measure one or more composition rows ordered like :attr:`components`."""
        components = self.components
        parameters = np.atleast_2d(np.asarray(parameters, dtype=float))
        if parameters.ndim != 2 or parameters.shape[1] != len(components):
            raise ValueError(f"parameters must have shape (n_samples, {len(components)}) for components {components}")
        if not np.isfinite(parameters).all():
            raise ValueError("compositions must be finite")

        rows_I, rows_I0, rows_dI, labels, in_hull = [], [], [], [], []
        q = None
        for row in parameters:
            label, inside = self.label_for(row)
            I, I0, dI, q = self.generate(label)
            rows_I.append(I)
            rows_I0.append(I0)
            rows_dI.append(dI)
            labels.append(int(label) if label.isdigit() else label)
            in_hull.append(inside)

        return xr.Dataset(
            data_vars={
                "composition": (("sample", "component"), parameters),
                "I": (("sample", "q"), np.stack(rows_I)),
                "I_noiseless": (("sample", "q"), np.stack(rows_I0)),
                "dI": (("sample", "q"), np.stack(rows_dI)),
                "labels": (("sample",), np.asarray(labels)),
                "in_hull": (("sample",), np.asarray(in_hull)),
            },
            coords={"sample": np.arange(parameters.shape[0]), "component": components, "q": q},
            attrs={"components": components, "components_dim": "component", "labels": "labels"},
        )

    def simple_expose(self, composition: dict) -> xr.Dataset:
        """Single-composition interface matching the original ``VirtualSAS_theory.simple_expose``.

        ``composition`` maps component name to value. The returned dataset has ``I``,
        ``I_noiseless`` and ``dI`` over ``q`` and the phase label as a string in
        ``ds.attrs["labels"]``.
        """
        components = self.components
        row = np.array([composition[c] for c in components], dtype=float)
        label, inside = self.label_for(row)
        I, I0, dI, q = self.generate(label)
        ds = xr.Dataset(
            data_vars={
                "I": (("q",), I),
                "I_noiseless": (("q",), I0),
                "dI": (("q",), dI),
                "composition": (("component",), row),
            },
            coords={"q": q, "component": components},
            attrs={
                "components": components,
                "components_dim": "component",
                "labels": str(label),
                "labels_dim": "label",
                "in_hull": bool(inside),
            },
        )
        return ds

    def status(self) -> list[str]:
        return [
            self.status_txt,
            f"Configurations Loaded={len(self.reference_data)}",
            f"Phases Traced={len(self.hulls)}",
            f"Noise Level={self.config['noise']}",
        ]


class ProteinGlycerolTemperaturePhaseMap(VirtualSASInstrument):
    """Virtual SAS instrument for the protein / temperature / glycerol phase map.

    Reproduces the AFL-bioformulations ``get_simulator`` configuration exactly: the
    ``bioformulation_phase_boundaries`` dataset packaged in ``AFL/double_agent/data``,
    three NIST SANS reference configurations, and ``power_law`` / ``sphere`` /
    ``polymer_excl_volume`` SasView models for phases 0 (solid-liquid), 1 (single
    phase) and 2 (liquid-liquid).

    ``measure`` expects composition rows ordered ``(protein, temperature, glycerol)``.
    """

    instrument_name = "Virtual SAS Instrument (protein/glycerol/temperature)"

    # Full merged dict: the fallback Driver reads only this class's ``defaults`` and the
    # AFL-automation Driver lets parent keys win, so the shared keys must be duplicated.
    defaults = {
        **VirtualSASInstrument.defaults,
        "boundary_dataset": "bioformulation_phase_boundaries",
        "component_limits": {
            "protein": (0.5, 75.0),
            "glycerol": (0.5, 10.0),
            "temperature": (0.5, 45.0),
        },
        "reference_configurations": ["SANS/low_q.ABS", "SANS/med_q.ABS", "SANS/high_q.ABS"],
        "sasview_models": {
            "0": {"model_name": "power_law", "model_kw": {"scale": 1e-7, "background": 1.0, "power": 4.0}},
            "1": {
                "model_name": "sphere",
                "model_kw": {"scale": 0.5, "background": 1.0, "sld": 1.0, "sld_solvent": 6.0, "radius": 10},
            },
            "2": {
                "model_name": "polymer_excl_volume",
                "model_kw": {"scale": 1.0, "background": 1.0, "rg": 100.0},
            },
        },
    }
