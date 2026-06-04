"""Tests for pipeline-op discovery in AFL.double_agent.AgentDriver."""

import importlib.util
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def _load_agentdriver_module_with_automation_stubs():
    """Load AgentDriver with minimal AFL.automation stubs required for import isolation."""
    driver_module = ModuleType("AFL.automation.APIServer.Driver")

    class StubDriver:
        @staticmethod
        def unqueued(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    driver_module.Driver = StubDriver

    utilities_module = ModuleType("AFL.automation.shared.utilities")
    utilities_module.mpl_plot_to_bytes = lambda *args, **kwargs: b""
    utilities_module.xarray_to_bytes = lambda *args, **kwargs: b""

    launcher_module = ModuleType("AFL.automation.shared.launcher")

    stub_modules = {
        "AFL.automation": ModuleType("AFL.automation"),
        "AFL.automation.APIServer": ModuleType("AFL.automation.APIServer"),
        "AFL.automation.APIServer.Driver": driver_module,
        "AFL.automation.shared": ModuleType("AFL.automation.shared"),
        "AFL.automation.shared.utilities": utilities_module,
        "AFL.automation.shared.launcher": launcher_module,
    }

    agentdriver_path = Path(__file__).resolve().parents[1] / "AFL" / "double_agent" / "AgentDriver.py"
    spec = importlib.util.spec_from_file_location("test_agentdriver_module", agentdriver_path)
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, stub_modules):
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return module


@pytest.mark.parametrize(
    "tensorflow_error",
    [
        ModuleNotFoundError("No module named 'tensorflow'", name="tensorflow"),
        PackageNotFoundError("tensorflow"),
    ],
    ids=["missing-module", "missing-package-metadata"],
)
def test_collect_pipeline_ops_skips_unavailable_tensorflow_dependency(tensorflow_error):
    agentdriver = _load_agentdriver_module_with_automation_stubs()
    package = SimpleNamespace(__path__=["fake-path"], __name__="AFL.double_agent")
    modinfos = [SimpleNamespace(name="TensorFlowExtrapolator"), SimpleNamespace(name="PyTorchExtrapolator")]
    pytorch_module = SimpleNamespace()

    def fake_import_module(name):
        if name == "AFL.double_agent":
            return package
        if name == "AFL.double_agent.TensorFlowExtrapolator":
            raise tensorflow_error
        if name == "AFL.double_agent.PyTorchExtrapolator":
            return pytorch_module
        raise AssertionError(f"Unexpected import: {name}")

    with (
        patch.object(agentdriver.importlib, "import_module", side_effect=fake_import_module),
        patch.object(agentdriver.pkgutil, "iter_modules", return_value=modinfos),
        patch.object(agentdriver.inspect, "getmembers", return_value=[]),
    ):
        ops = agentdriver._collect_pipeline_ops()

    assert ops == []


def test_collect_pipeline_ops_skips_runtime_wrapped_optional_dependency_errors():
    agentdriver = _load_agentdriver_module_with_automation_stubs()
    package = SimpleNamespace(__path__=["fake-path"], __name__="AFL.double_agent")
    modinfos = [SimpleNamespace(name="AmplitudePhaseDistance")]

    def fake_import_module(name):
        if name == "AFL.double_agent":
            return package
        if name == "AFL.double_agent.AmplitudePhaseDistance":
            raise RuntimeError(
                "ImportError encountered: No module named 'apdist'\n"
                "To use amplitude-distance as a similarity measure, please install:\n"
                "pip install git+https://github.com/kiranvad/Amplitude-Phase-Distance"
            )
        raise AssertionError(f"Unexpected import: {name}")

    with (
        patch.object(agentdriver.importlib, "import_module", side_effect=fake_import_module),
        patch.object(agentdriver.pkgutil, "iter_modules", return_value=modinfos),
        patch.object(agentdriver.inspect, "getmembers", return_value=[]),
    ):
        ops = agentdriver._collect_pipeline_ops()

    assert ops == []


def test_collect_pipeline_ops_raises_for_internal_module_errors():
    agentdriver = _load_agentdriver_module_with_automation_stubs()
    package = SimpleNamespace(__path__=["fake-path"], __name__="AFL.double_agent")
    modinfos = [SimpleNamespace(name="BrokenModule")]

    def fake_import_module(name):
        if name == "AFL.double_agent":
            return package
        if name == "AFL.double_agent.BrokenModule":
            raise ModuleNotFoundError("No module named 'AFL.double_agent.missing_internal'", name="AFL.double_agent.missing_internal")
        raise AssertionError(f"Unexpected import: {name}")

    with (
        patch.object(agentdriver.importlib, "import_module", side_effect=fake_import_module),
        patch.object(agentdriver.pkgutil, "iter_modules", return_value=modinfos),
    ):
        with pytest.raises(ModuleNotFoundError, match="AFL.double_agent.missing_internal"):
            agentdriver._collect_pipeline_ops()


def test_optional_dependency_detection_does_not_bypass_automation_import_errors():
    agentdriver = _load_agentdriver_module_with_automation_stubs()

    exc = ModuleNotFoundError(
        "No module named 'AFL.automation.shared.utilities'",
        name="AFL.automation.shared.utilities",
    )

    assert agentdriver._is_optional_dependency_failure(exc) is False
