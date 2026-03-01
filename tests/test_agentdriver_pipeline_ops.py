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
