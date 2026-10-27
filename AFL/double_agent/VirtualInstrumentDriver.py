"""Serve any virtual instrument of :mod:`AFL.double_agent.VirtualInstruments` as an AFL-automation driver.

An AFL-automation instrument runs as an ``APIServer`` whose queue executes the public
methods of a ``Driver``. :class:`VirtualInstrumentDriver` is such a driver: it builds one
virtual instrument from its configuration and forwards every public method of that
instrument, so each becomes a task of the client/server model (``measure``, ``expose``,
``status``, ``locate``, ...). The result of each task is written to Tiled by the server
exactly as for a hardware driver, under the ``driver_name`` of the instrument.

Launch it like any other AFL driver, selecting the instrument through a JSON config file::

    python -m AFL.double_agent.VirtualInstrumentDriver --config phase_map.json

where ``phase_map.json`` holds the driver configuration::

    {
        "instrument": "ProteinGlycerolTemperaturePhaseMap",
        "instrument_overrides": {"noise": 0.05, "random_seed": 11}
    }

``instrument`` is the name of a class in :mod:`AFL.double_agent.VirtualInstruments` or the
dotted path of a class elsewhere; ``instrument_overrides`` are the configuration overrides
passed to that class. Without ``--config`` the :class:`GaussianVirtualInstrument` is served.

The server takes the name of the instrument class: it is registered under that name in the
``"ports"`` of ``~/.afl/config.json`` (``_DEFAULT_PORT`` on first launch), the Tiled entries
carry it as ``driver_name`` and the driver configuration is snapshotted to
``~/.afl/configs/<InstrumentName>.config.json``.

From a client, the instrument's methods are plain tasks::

    client = Client("localhost", port=5052)
    client.login("user")
    client.enqueue(task_name="expose", protein=40.0, glycerol=5.0, temperature=20.0, interactive=True)
    client.enqueue(task_name="instrument_tasks", interactive=True)["return_val"]
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

from AFL.double_agent import VirtualInstruments as _virtual_instruments
from AFL.double_agent._automation_compat import Driver

_DEFAULT_PORT = 5052

#: Attributes the server assigns to the driver that the instrument needs as well.
_SHARED_ATTRIBUTES = ("app", "data", "dropbox")


def available_instruments() -> list[str]:
    """Names of the virtual instrument classes defined in :mod:`AFL.double_agent.VirtualInstruments`."""
    names = []
    for name, obj in vars(_virtual_instruments).items():
        if (
            inspect.isclass(obj)
            and obj.__module__ == _virtual_instruments.__name__
            and callable(getattr(obj, "measure", None))
        ):
            names.append(name)
    return sorted(names)


def resolve_instrument_class(spec: str | type) -> type:
    """Return the virtual instrument class named by ``spec``.

    ``spec`` is a class, the name of a class in :mod:`AFL.double_agent.VirtualInstruments`
    or the dotted path ``package.module.ClassName`` of a class elsewhere.
    """
    if inspect.isclass(spec):
        cls = spec
    elif isinstance(spec, str) and spec:
        if "." in spec:
            module_name, _, class_name = spec.rpartition(".")
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                raise ValueError(f"cannot import module {module_name!r} of instrument {spec!r}") from exc
            cls = getattr(module, class_name, None)
        else:
            cls = getattr(_virtual_instruments, spec, None)
        if cls is None:
            raise ValueError(
                f"unknown virtual instrument {spec!r}; available instruments are {available_instruments()} "
                "or a dotted class path"
            )
    else:
        raise TypeError(f"instrument must be a class or a class name, got {spec!r}")

    if not inspect.isclass(cls) or not callable(getattr(cls, "measure", None)):
        raise ValueError(f"{spec!r} is not a virtual instrument class: it has no 'measure' method")
    return cls


def instrument_spec(cls: type) -> str:
    """The shortest ``instrument`` string that resolves back to ``cls``."""
    if cls.__module__ == _virtual_instruments.__name__:
        return cls.__name__
    return f"{cls.__module__}.{cls.__qualname__}"


def _shared_attribute(name: str) -> property:
    """A driver attribute whose value is mirrored onto the wrapped instrument."""
    slot = f"_{name}"

    def fget(self):
        return self.__dict__.get(slot)

    def fset(self, value):
        self.__dict__[slot] = value
        instrument = self.__dict__.get("instrument")
        if instrument is not None:
            setattr(instrument, name, value)

    return property(fget, fset, doc=f"``{name}`` of the server, shared with the instrument.")


class VirtualInstrumentDriver(Driver):
    """AFL-automation driver that serves one configured virtual instrument.

    Parameters
    ----------
    overrides : dict, optional
        Driver configuration. ``instrument`` selects the instrument class (see
        :func:`resolve_instrument_class`) and ``instrument_overrides`` is passed to it as
        its own configuration overrides.
    afl_home : str or Path, optional
        Root for the driver's persistent configuration, ``~/.afl`` by default. The
        instrument keeps its own configuration in the ``virtual_instruments`` directory
        below it so that the two files never collide.
    instrument : object or type, optional
        An instrument instance to serve as is, or an instrument class to build, instead of
        the one named in the configuration.

    Every public method of the instrument that the driver does not define itself is
    reachable as an attribute of the driver and therefore as a task of the server. The
    driver's own ``status``, ``pre_execute`` and ``post_execute`` delegate to the
    instrument's when it overrides them.
    """

    defaults = {
        "instrument": "GaussianVirtualInstrument",
        "instrument_overrides": {},
    }

    app = _shared_attribute("app")
    data = _shared_attribute("data")
    dropbox = _shared_attribute("dropbox")

    def __init__(
        self,
        overrides: dict | None = None,
        afl_home: str | Path | None = None,
        instrument: object | type | None = None,
    ) -> None:
        overrides = dict(overrides or {})
        if instrument is None:
            instrument_cls = resolve_instrument_class(overrides.get("instrument", self.gather_defaults()["instrument"]))
        elif inspect.isclass(instrument):
            instrument_cls = instrument
        else:
            instrument_cls = type(instrument)
        overrides["instrument"] = instrument_spec(instrument_cls)

        self.app = None
        _virtual_instruments._driver_init(self, instrument_cls.__name__, overrides, afl_home)

        if instrument is None or inspect.isclass(instrument):
            instrument = self._build_instrument(instrument_cls)
        self.instrument = instrument
        for name in _SHARED_ATTRIBUTES:
            setattr(instrument, name, getattr(self, name))

    @classmethod
    def gather_defaults(cls) -> dict:
        """Merge the ``defaults`` of the class hierarchy, letting subclasses win."""
        defaults = {}
        for parent in reversed(cls.__mro__):
            defaults.update(getattr(parent, "defaults", {}) or {})
        return defaults

    @classmethod
    def for_instrument(cls, instrument: str | type) -> type["VirtualInstrumentDriver"]:
        """Return a subclass named after ``instrument`` that serves it by default.

        The AFL launcher names the server, its port entry and its configuration snapshot
        after the driver class, so the launcher uses this to give each instrument its own.
        """
        instrument_cls = resolve_instrument_class(instrument)
        return type(
            instrument_cls.__name__,
            (cls,),
            {
                "defaults": {**cls.defaults, "instrument": instrument_spec(instrument_cls)},
                "__module__": cls.__module__,
                "__doc__": f"{cls.__name__} serving {instrument_spec(instrument_cls)}.",
            },
        )

    def _build_instrument(self, instrument_cls: type):
        params = inspect.signature(instrument_cls).parameters
        kwargs = {}
        if "overrides" in params:
            kwargs["overrides"] = dict(self.config.get("instrument_overrides") or {})
        path = getattr(self, "path", None)
        if "afl_home" in params and path is not None:
            kwargs["afl_home"] = Path(path) / "virtual_instruments"
        return instrument_cls(**kwargs)

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails, i.e. for names the driver itself lacks.
        if name.startswith("_") or name == "instrument":
            raise AttributeError(name)
        instrument = self.__dict__.get("instrument")
        if instrument is None:
            raise AttributeError(name)
        return getattr(instrument, name)

    def _instrument_overrides(self, name: str) -> bool:
        """Whether the instrument defines ``name`` itself rather than inheriting the Driver one."""
        own = inspect.getattr_static(type(self.instrument), name, None)
        return own is not None and own is not inspect.getattr_static(Driver, name, None)

    def status(self) -> list:
        if self._instrument_overrides("status"):
            return list(self.instrument.status())
        return [f"Serving {self.config['instrument']}"]

    def pre_execute(self, **kwargs) -> None:
        if self._instrument_overrides("pre_execute"):
            self.instrument.pre_execute(**kwargs)

    def post_execute(self, **kwargs) -> None:
        if self._instrument_overrides("post_execute"):
            self.instrument.post_execute(**kwargs)

    def instrument_info(self) -> dict:
        """Describe the served instrument: class, module and its configuration."""
        instrument_cls = type(self.instrument)
        config = getattr(self.instrument, "config", None)
        if config is not None and not isinstance(config, dict):
            config = dict(config.config) if hasattr(config, "config") else dict(config)
        return {
            "instrument": instrument_spec(instrument_cls),
            "class": instrument_cls.__name__,
            "module": instrument_cls.__module__,
            "instrument_name": getattr(self.instrument, "__instrument_name__", instrument_cls.__name__),
            "config": config,
            "tasks": sorted(self.instrument_tasks()),
        }

    def instrument_tasks(self) -> dict[str, dict]:
        """Public methods of the instrument served as tasks, with their signatures.

        Methods every driver has (``set_config``, ``set_sample``, ...) are left out; they
        are served by the driver itself.
        """
        tasks = {}
        for name, member in inspect.getmembers(type(self.instrument), callable):
            if name.startswith("_") or name in ("status",) or isinstance(member, type):
                continue
            if getattr(Driver, name, None) is not None and not self._instrument_overrides(name):
                continue
            try:
                signature = str(inspect.signature(getattr(self.instrument, name)))
            except (TypeError, ValueError):
                signature = "(...)"
            doc = inspect.getdoc(member) or ""
            tasks[name] = {"signature": signature, "doc": doc.splitlines()[0] if doc else ""}
        tasks["status"] = {"signature": "()", "doc": "Status lines of the instrument."}
        return tasks

    def get_instrument_config(self) -> dict:
        """The current configuration of the instrument."""
        config = self.instrument.config
        return dict(config.config) if hasattr(config, "config") else dict(config)

    def set_instrument_config(self, **kwargs) -> dict:
        """Update the instrument's configuration and remember it in ``instrument_overrides``.

        The remembered overrides are applied again when the server is restarted from the
        same driver configuration.
        """
        if hasattr(self.instrument, "set_config"):
            self.instrument.set_config(**kwargs)
        else:
            self.instrument.config.update(kwargs)
        overrides = dict(self.config.get("instrument_overrides") or {})
        overrides.update(kwargs)
        self.config["instrument_overrides"] = overrides
        return self.get_instrument_config()


def _launch_config_path(argv: list[str]) -> Path | None:
    """The ``--config`` path of the launcher's command line, if given."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    known, _ = parser.parse_known_args(argv)
    return known.config


def _launch_instrument_class(argv: list[str]) -> type:
    """The instrument class the launcher's ``--config`` file (or the default) selects."""
    spec = VirtualInstrumentDriver.defaults["instrument"]
    config_path = _launch_config_path(argv)
    if config_path is not None:
        try:
            launch_config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Unable to read --config file {config_path}: {exc}")
        if not isinstance(launch_config, dict):
            raise SystemExit("--config must contain a JSON object")
        spec = launch_config.get("instrument", spec)
    return resolve_instrument_class(spec)


if __name__ == "__main__":
    # The launcher names the server, its port entry and its configuration snapshot after
    # the class it finds under the module name. Point it at a subclass named after the
    # instrument so each instrument gets its own.
    _instrument_cls = _launch_instrument_class(sys.argv[1:])
    _OVERRIDE_MAIN_MODULE_NAME = _instrument_cls.__name__
    globals()[_OVERRIDE_MAIN_MODULE_NAME] = VirtualInstrumentDriver.for_instrument(_instrument_cls)
    from AFL.automation.shared.launcher import *  # noqa: F401,F403
