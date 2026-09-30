"""Virtual instruments used in AFL-agent examples and local simulations."""

from __future__ import annotations

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
        Driver.__init__(
            self,
            name="GaussianVirtualInstrument",
            defaults=self.gather_defaults(),
            overrides=overrides,
            afl_home=afl_home,
        )
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
