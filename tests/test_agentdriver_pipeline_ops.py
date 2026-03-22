from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import AFL.double_agent._automation_compat as automation_compat
import AFL.double_agent.AgentDriver as agent_driver


def test_parse_strict_flag():
    assert agent_driver._parse_strict_flag(True) is True
    assert agent_driver._parse_strict_flag(False) is False
    assert agent_driver._parse_strict_flag("true") is True
    assert agent_driver._parse_strict_flag("1") is True
    assert agent_driver._parse_strict_flag("yes") is True
    assert agent_driver._parse_strict_flag("on") is True
    assert agent_driver._parse_strict_flag("false") is False
    assert agent_driver._parse_strict_flag(None) is False


def test_get_pipeline_ops_uses_memory_cache(monkeypatch):
    monkeypatch.setattr(agent_driver, "_PIPELINE_OPS_MEM_CACHE", None)
    monkeypatch.setattr(agent_driver, "_candidate_module_files", lambda: [])
    monkeypatch.setattr(agent_driver, "_module_signature", lambda module_files: "sig-1")
    monkeypatch.setattr(agent_driver, "_load_disk_cache", lambda expected_signature: None)

    call_count = {"count": 0}

    def fake_collect(module_files, strict=False):
        call_count["count"] += 1
        return ([{"name": "OpA"}], [])

    monkeypatch.setattr(agent_driver, "_collect_pipeline_ops", fake_collect)
    monkeypatch.setattr(agent_driver, "_save_disk_cache", lambda payload: None)

    first = agent_driver.get_pipeline_ops()
    second = agent_driver.get_pipeline_ops()

    assert first["ops"] == [{"name": "OpA"}]
    assert first["cache"]["source"] == "fresh"
    assert second["cache"]["source"] == "memory"
    assert call_count["count"] == 1


def test_get_pipeline_ops_uses_disk_cache(monkeypatch):
    monkeypatch.setattr(agent_driver, "_PIPELINE_OPS_MEM_CACHE", None)
    monkeypatch.setattr(agent_driver, "_candidate_module_files", lambda: [])
    monkeypatch.setattr(agent_driver, "_module_signature", lambda module_files: "sig-2")
    monkeypatch.setattr(
        agent_driver,
        "_load_disk_cache",
        lambda expected_signature: {
            "ops": [{"name": "DiskOp"}],
            "warnings": [{"module": "A", "stage": "import", "error_type": "E", "message": "m"}],
            "generated_at": "2026-03-01T00:00:00+00:00",
            "signature": "sig-2",
        },
    )
    monkeypatch.setattr(
        agent_driver,
        "_collect_pipeline_ops",
        lambda module_files, strict=False: (_ for _ in ()).throw(AssertionError("collect should not run")),
    )

    result = agent_driver.get_pipeline_ops()

    assert result["ops"] == [{"name": "DiskOp"}]
    assert result["warnings"][0]["stage"] == "import"
    assert result["cache"]["source"] == "disk"


def test_get_pipeline_ops_strict_skips_cache(monkeypatch):
    monkeypatch.setattr(
        agent_driver,
        "_PIPELINE_OPS_MEM_CACHE",
        {
            "ops": [{"name": "Cached"}],
            "warnings": [],
            "cache": {"source": "memory", "generated_at": "old", "signature": "sig-3", "duration_ms": 0},
            "signature": "sig-3",
        },
    )
    monkeypatch.setattr(agent_driver, "_candidate_module_files", lambda: [])
    monkeypatch.setattr(agent_driver, "_module_signature", lambda module_files: "sig-3")

    call_count = {"count": 0}

    def fake_collect(module_files, strict=False):
        call_count["count"] += 1
        assert strict is True
        return ([{"name": "FreshStrict"}], [])

    monkeypatch.setattr(agent_driver, "_collect_pipeline_ops", fake_collect)
    monkeypatch.setattr(agent_driver, "_save_disk_cache", lambda payload: None)

    result = agent_driver.get_pipeline_ops(strict=True)

    assert result["ops"] == [{"name": "FreshStrict"}]
    assert result["cache"]["source"] == "fresh"
    assert call_count["count"] == 1


def test_double_agent_driver_static_dirs_point_to_apps():
    static_dirs = agent_driver.DoubleAgentDriver.static_dirs

    assert "apps/pipeline_builder/js" in static_dirs
    assert "apps/pipeline_builder/img" in static_dirs
    assert "apps/pipeline_builder/css" in static_dirs
    assert "apps/input_builder/js" in static_dirs
    assert "apps/input_builder/css" in static_dirs

    assert "apps/pipeline_builder/js" in str(static_dirs["apps/pipeline_builder/js"])
    assert "apps/pipeline_builder/img" in str(static_dirs["apps/pipeline_builder/img"])
    assert "apps/pipeline_builder/css" in str(static_dirs["apps/pipeline_builder/css"])
    assert "apps/input_builder/js" in str(static_dirs["apps/input_builder/js"])
    assert "apps/input_builder/css" in str(static_dirs["apps/input_builder/css"])


def test_web_app_mixin_renders_builder_html():
    pipeline_html = agent_driver.DoubleAgentDriver._render_pipeline_builder_html()
    input_html = agent_driver.DoubleAgentDriver._render_input_builder_html()

    assert "<!DOCTYPE html>" in pipeline_html
    assert "<!DOCTYPE html>" in input_html
    assert "<title>Pipeline Builder</title>" in pipeline_html
    assert "<title>Input Builder</title>" in input_html


def test_setup_app_links_sets_builder_links():
    driver = agent_driver.DoubleAgentDriver.__new__(agent_driver.DoubleAgentDriver)

    driver.useful_links = None
    driver.setup_app_links()
    assert driver.useful_links == {
        "Pipeline Builder": "/pipeline_builder",
        "Input Builder": "/input_builder",
    }

    driver.useful_links = {"Existing": "/existing"}
    driver.setup_app_links()
    assert driver.useful_links["Existing"] == "/existing"
    assert driver.useful_links["Pipeline Builder"] == "/pipeline_builder"
    assert driver.useful_links["Input Builder"] == "/input_builder"


def test_app_backend_methods_are_mixin_owned():
    assert agent_driver.DoubleAgentDriver.plot_pipeline.__qualname__.startswith("AgentWebAppMixin.")
    assert agent_driver.DoubleAgentDriver.get_tiled_input_config.__qualname__.startswith("AgentWebAppMixin.")
    assert agent_driver.DoubleAgentDriver.pipeline_ops.__qualname__.startswith("AgentWebAppMixin.")
    assert agent_driver.DoubleAgentDriver.assemble_input_from_tiled.__qualname__.startswith("AgentWebAppMixin.")


def test_fallback_driver_exposes_tiled_helpers_for_entry_lookup():
    driver = automation_compat.FallbackDriver()
    item = SimpleNamespace(metadata={"attrs": {"task_name": "measure_scattering"}})
    driver._tiled_client = {
        driver.TILED_RUN_DOCUMENTS_NODE: {
            "QD-123": item,
        }
    }

    normalized_entry_id, selected_item = driver._get_tiled_run_document_item("run_documents/QD-123")

    assert hasattr(automation_compat.FallbackDriver, "_get_tiled_client")
    assert hasattr(driver, "_get_tiled_client")
    assert normalized_entry_id == "QD-123"
    assert selected_item is item


@pytest.mark.parametrize(
    ("supplied_entry_id", "expected_entry_id"),
    [
        ("QD-123", "QD-123"),
        ("run_documents/QD-123", "QD-123"),
    ],
)
def test_test_fetch_entry_accepts_run_document_entry_ids(monkeypatch, supplied_entry_id, expected_entry_id):
    driver = agent_driver.DoubleAgentDriver.__new__(agent_driver.DoubleAgentDriver)

    dataset = xr.Dataset(
        {"I": ("q", [1.0, 2.0])},
        coords={"q": [0.1, 0.2]},
    )
    item = SimpleNamespace(metadata={"attrs": {"task_name": "measure_scattering"}})
    lookup_calls = []

    monkeypatch.setattr(driver, "_get_tiled_client", lambda: object())
    monkeypatch.setattr(
        driver,
        "_get_tiled_run_document_item",
        lambda entry_id: (lookup_calls.append(entry_id) or ("QD-123", item)),
    )
    monkeypatch.setattr(driver, "_read_tiled_item", lambda selected_item: dataset if selected_item is item else None)

    result = driver.test_fetch_entry(supplied_entry_id)

    assert result["status"] == "success"
    assert result["entry_id"] == expected_entry_id
    assert result["metadata"]["attrs"]["task_name"] == "measure_scattering"
    assert result["dims"] == {"q": 2}
    assert result["data_vars"] == ["I"]
    assert lookup_calls == [supplied_entry_id]


def test_test_fetch_entry_falls_back_to_direct_client_lookup(monkeypatch):
    driver = agent_driver.DoubleAgentDriver.__new__(agent_driver.DoubleAgentDriver)

    dataset = xr.Dataset({"I": ("q", [1.0])}, coords={"q": [0.1]})
    item = SimpleNamespace(
        metadata={"attrs": {"sample_uuid": "SAM-001"}},
        read=lambda optimize_wide_table=False: dataset,
    )
    client = {"legacy-entry": item}

    monkeypatch.setattr(driver, "_get_tiled_client", lambda: client)
    monkeypatch.setattr(driver, "_get_tiled_run_document_item", None, raising=False)
    monkeypatch.setattr(driver, "_read_tiled_item", None, raising=False)

    result = driver.test_fetch_entry("legacy-entry")

    assert result["status"] == "success"
    assert result["entry_id"] == "legacy-entry"
    assert result["metadata"]["attrs"]["sample_uuid"] == "SAM-001"
    assert result["dims"] == {"q": 1}
    assert result["data_vars"] == ["I"]


def test_predict_sanitizes_object_dtype_datasets():
    driver = agent_driver.DoubleAgentDriver.__new__(agent_driver.DoubleAgentDriver)
    driver.config = {"save_path": "/tmp", "tiled_input_groups": []}
    driver.input = xr.Dataset(
        data_vars={
            "raw_text": ("sample", np.asarray(["alpha", "beta"], dtype=object)),
        },
        coords={
            "sample": [0, 1],
            "label": ("sample", np.asarray(["s1", "s2"], dtype=object)),
        },
    )
    driver.pipeline = SimpleNamespace(
        calculate=lambda dataset: dataset.assign(
            result_text=("sample", np.asarray(["left", "right"], dtype=object))
        )
    )
    driver.last_results = None
    driver.assemble_input_from_tiled = lambda: None
    driver.deposit_obj = lambda *args, **kwargs: None

    result = driver.predict()

    assert driver.input["raw_text"].dtype.kind != "O"
    assert driver.input.coords["label"].dtype.kind != "O"
    assert driver.last_results["raw_text"].dtype.kind != "O"
    assert driver.last_results["result_text"].dtype.kind != "O"
    assert driver.last_results.coords["label"].dtype.kind != "O"
    assert result["raw_text"].values.tolist() == ["alpha", "beta"]
    assert result["result_text"].values.tolist() == ["left", "right"]
