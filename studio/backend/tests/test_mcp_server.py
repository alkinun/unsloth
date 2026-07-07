# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the standalone Studio MCP server (``studio/backend/mcp_server``).

Two layers:
  * unit tests call the tool handler functions directly (with the backend
    singletons mocked, so no GPU/subprocess is needed);
  * integration tests drive a built FastMCP server through its in-memory
    client, exercising the real registration + schema path.
"""

import asyncio

import pytest


# ── gating & auth helpers ─────────────────────────────────────────────


def test_resolve_groups_defaults_to_all():
    from mcp_server.tools import FEATURE_GROUPS, resolve_groups

    assert resolve_groups() == FEATURE_GROUPS
    assert resolve_groups(None, None) == FEATURE_GROUPS


def test_resolve_groups_enable_subset_preserves_order():
    from mcp_server.tools import resolve_groups

    assert resolve_groups(["data", "models"]) == ["models", "data"]
    assert resolve_groups(["recipe", "train"]) == ["train", "recipe"]


def test_resolve_groups_disable_removes_from_all():
    from mcp_server.tools import resolve_groups

    assert resolve_groups(disabled=["train"]) == ["models", "data", "export", "recipe"]


def test_resolve_groups_enable_then_disable():
    from mcp_server.tools import resolve_groups

    # disable wins over enable
    assert resolve_groups(["models", "data", "train"], ["train"]) == ["models", "data"]


def test_resolve_groups_unknown_raises():
    from mcp_server.tools import resolve_groups

    with pytest.raises(ValueError, match="Unknown feature group"):
        resolve_groups(["bogus"])
    with pytest.raises(ValueError, match="Unknown feature group"):
        resolve_groups(disabled=["nope"])


def test_resolve_secret_prefers_override_then_env(monkeypatch):
    from mcp_server.auth import resolve_hf_token, resolve_secret

    monkeypatch.setenv("HF_TOKEN", "envval")
    assert resolve_secret("HF_TOKEN") == "envval"
    assert resolve_secret("HF_TOKEN", "explicit") == "explicit"
    assert resolve_secret("HF_TOKEN", "  ") == "envval"  # blank override ignored
    monkeypatch.delenv("HF_TOKEN")
    assert resolve_secret("HF_TOKEN") is None

    monkeypatch.setenv("HF_TOKEN", "hf-env")
    assert resolve_hf_token() == "hf-env"
    assert resolve_hf_token("override") == "override"
    # HUGGING_FACE_HUB_TOKEN fallback
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hub-env")
    assert resolve_hf_token() == "hub-env"


# ── models tools (read-only, real implementation) ─────────────────────


def test_models_list_returns_curated_catalog():
    from mcp_server.tools.models import models_list

    result = models_list()
    assert result["count"] == len(result["models"])
    assert result["count"] > 0
    first = result["models"][0]
    assert {"id", "name", "aliases", "config_file"} <= set(first)
    assert isinstance(first["aliases"], list) and first["aliases"]


def test_models_list_sorted_and_stable():
    from mcp_server.tools.models import models_list

    ids = [m["id"].lower() for m in models_list()["models"]]
    assert ids == sorted(ids)


def test_models_get_config_returns_yaml_defaults():
    from mcp_server.tools.models import models_get_config

    result = models_get_config("unsloth/llama-3-8b-bnb-4bit")
    assert result["model_name"] == "unsloth/llama-3-8b-bnb-4bit"
    assert isinstance(result["config"], dict)


def test_models_get_config_unknown_model_falls_back_to_default():
    # load_model_defaults falls back to default.yaml for unknown models.
    from mcp_server.tools.models import models_get_config

    result = models_get_config("this/does-not-exist-xyz")
    assert result["model_name"] == "this/does-not-exist-xyz"
    assert isinstance(result["config"], dict)
    assert "training" in result["config"]  # default.yaml fallback


# ── data tools ────────────────────────────────────────────────────────


def test_data_register_and_list_round_trip(tmp_path, monkeypatch):
    from mcp_server.tools import data as data_tools

    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr("hub.services.datasets.local.DATASET_UPLOAD_DIR", upload_dir, raising=True)

    src = tmp_path / "my.csv"
    src.write_text("text,label\nhello,1\nworld,0\n")

    result = data_tools.data_register(str(src))
    assert result["success"] is True
    assert result["filename"] == "my.csv"
    stored_name = result["stored_path"].split("/")[-1]
    assert (upload_dir / stored_name).is_file()

    listed = data_tools.data_list_local()
    ids = [d["id"] for d in listed["datasets"]]
    assert result["stored_path"].split("/")[-1] in ids or any(
        result["stored_path"] == d["path"] for d in listed["datasets"]
    )


def test_data_register_rejects_bad_extension(tmp_path):
    from mcp_server.tools.data import data_register

    src = tmp_path / "data.exe"
    src.write_text("nope")
    result = data_register(str(src))
    assert result["success"] is False
    assert "Unsupported file type" in result["error"]


def test_data_register_rejects_missing_file():
    from mcp_server.tools.data import data_register

    result = data_register("/nonexistent/path/to/file.csv")
    assert result["success"] is False
    assert "Not a file" in result["error"]


def test_data_check_format_uses_hub_request_schema(monkeypatch):
    # Regression: data_check_format must build the *hub* CheckFormatRequest
    # (hub.schemas.datasets, which has prefer_local_cache), not the legacy
    # models.datasets one -- otherwise check_format_response raises
    # AttributeError on request.prefer_local_cache.
    import hub.services.datasets.formatting as formatting_mod
    from hub.schemas.datasets import (
        CheckFormatRequest as HubCheckFormatRequest,
        CheckFormatResponse,
    )

    captured: dict = {}

    def _fake_check(request, hf_token=None):
        captured["request"] = request
        return CheckFormatResponse(
            requires_manual_mapping=False, detected_format="alpaca", columns=["instruction"]
        )

    monkeypatch.setattr(formatting_mod, "check_format_response", _fake_check)

    from mcp_server.tools.data import data_check_format

    # Should not raise; the constructed request must carry prefer_local_cache.
    data_check_format(dataset_name="some/dataset", is_vlm=True, train_split="train")
    request = captured["request"]
    assert isinstance(request, HubCheckFormatRequest)
    assert hasattr(request, "prefer_local_cache")
    assert request.dataset_name == "some/dataset"
    assert request.is_vlm is True


def test_data_check_format_surfaces_errors_cleanly():
    # A malformed dataset name must yield a clean error dict, never an exception.
    from mcp_server.tools.data import data_check_format

    result = data_check_format(dataset_name="<<<not-a-real-dataset>>>")
    assert result["success"] is False
    assert "error" in result


# ── training tools (mocked backend) ───────────────────────────────────


class _FakeTrainingProgress:
    def asdict(self):
        return {
            "epoch": 0,
            "step": 0,
            "loss": None,
            "status_message": "Ready to train",
            "is_training": False,
            "is_completed": False,
            "error": None,
        }


class _FakeTrainer:
    def __init__(self, progress):
        self._progress = progress

    def get_training_progress(self):
        return self._progress


class _FakeTrainingBackend:
    def __init__(self, *, active=False, start_succeeds=True):
        self._active = active
        self._start_succeeds = start_succeeds
        self.current_job_id = "job_existing" if active else None
        self.stop_called_with = None
        from core.training.training import TrainingProgress

        self._progress = TrainingProgress(status_message="Ready to train")
        self.trainer = _FakeTrainer(self._progress)

    def is_training_active(self):
        return self._active

    def start_training(self, job_id, **kwargs):
        self.last_job_id = job_id
        self.last_kwargs = kwargs
        return self._start_succeeds

    def stop_training(self, save=True):
        self.stop_called_with = save
        self._active = False


@pytest.fixture
def _patch_training_backend(monkeypatch):
    def _install(backend):
        import core.training.training as mod

        monkeypatch.setattr(mod, "get_training_backend", lambda: backend)
        return backend

    return _install


def test_train_start_rejects_when_already_active(_patch_training_backend):
    from mcp_server.tools.train import train_start

    _patch_training_backend(_FakeTrainingBackend(active=True))
    result = train_start(
        model_name="unsloth/llama-3-8b-bnb-4bit", hf_dataset="yahma/alpaca-cleaned"
    )
    assert result["success"] is False
    assert "already in progress" in result["error"]


def test_train_start_validates_and_starts(_patch_training_backend):
    from mcp_server.tools.train import train_start

    backend = _patch_training_backend(_FakeTrainingBackend(start_succeeds=True))
    result = train_start(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        hf_dataset="yahma/alpaca-cleaned",
        format_type="alpaca",
        num_epochs=1,
    )
    assert result["success"] is True
    assert result["status"] == "queued"
    assert result["job_id"].startswith("job_")
    # The job_id generated by the tool is what gets passed to the backend.
    assert backend.last_job_id == result["job_id"]
    assert backend.last_kwargs["model_name"] == "unsloth/llama-3-8b-bnb-4bit"
    assert backend.last_kwargs["subject"] == "mcp"
    assert backend.last_kwargs["hf_dataset"] == "yahma/alpaca-cleaned"


def test_train_start_invalid_training_type_returns_error(_patch_training_backend):
    from mcp_server.tools.train import train_start

    _patch_training_backend(_FakeTrainingBackend())
    result = train_start(model_name="unsloth/llama-3-8b-bnb-4bit", training_type="Bogus")
    assert result["success"] is False
    assert "error" in result


def test_train_start_failure_surfaces_progress_error(_patch_training_backend):
    from mcp_server.tools.train import train_start

    _patch_training_backend(_FakeTrainingBackend(start_succeeds=False))
    result = train_start(
        model_name="unsloth/llama-3-8b-bnb-4bit", hf_dataset="yahma/alpaca-cleaned"
    )
    assert result["success"] is False
    assert "status" in result and result["status"] == "error"


def test_train_status_reports_active_and_progress(_patch_training_backend):
    from mcp_server.tools.train import train_status

    _patch_training_backend(_FakeTrainingBackend(active=True))
    result = train_status()
    assert result["is_active"] is True
    assert result["job_id"] == "job_existing"
    assert "loss" in result["progress"]


def test_train_stop_when_idle(_patch_training_backend):
    from mcp_server.tools.train import train_stop

    backend = _patch_training_backend(_FakeTrainingBackend(active=False))
    result = train_stop()
    assert result["status"] == "idle"
    assert backend.stop_called_with is None


def test_train_stop_when_active(_patch_training_backend):
    from mcp_server.tools.train import train_stop

    backend = _patch_training_backend(_FakeTrainingBackend(active=True))
    result = train_stop(save=False)
    assert result["status"] == "stopping"
    assert backend.stop_called_with is False


# ── export tools (mocked backend) ─────────────────────────────────────


class _FakeExportBackend:
    def __init__(self):
        self.current_checkpoint = "/ckpt/last"
        self.is_vision = False
        self.is_peft = True
        self._last_op = {
            "kind": "export_merged",
            "status": "success",
            "output_path": "/out",
            "error": None,
        }
        self.cancelled = False
        self.cleaned = False

    def is_export_active(self):
        return False

    def get_active_op_kind(self):
        return None

    def get_last_op(self):
        return self._last_op

    def scan_checkpoints(self, outputs_dir=None):
        return [("mymodel", [("checkpoint-100", "/p/ckpt-100", 0.5)], {"foo": "bar"})]

    def cancel_export(self):
        self.cancelled = True
        return True

    def cleanup_memory(self):
        self.cleaned = True
        return True


@pytest.fixture
def _patch_export_backend(monkeypatch):
    def _install(backend):
        import core.export as mod

        monkeypatch.setattr(mod, "get_export_backend", lambda: backend)
        return backend

    return _install


def test_export_status_serializes_state(_patch_export_backend):
    from mcp_server.tools.export import export_status

    _patch_export_backend(_FakeExportBackend())
    result = export_status()
    assert result["current_checkpoint"] == "/ckpt/last"
    assert result["is_peft"] is True
    assert result["is_export_active"] is False
    assert result["last_op"]["kind"] == "export_merged"
    assert result["last_op"]["output_path"] == "/out"


def test_export_list_checkpoints_shapes_output(_patch_export_backend):
    from mcp_server.tools.export import export_list_checkpoints

    _patch_export_backend(_FakeExportBackend())
    result = export_list_checkpoints()
    assert result["runs"][0]["model_name"] == "mymodel"
    assert result["runs"][0]["checkpoints"][0]["loss"] == 0.5


def test_export_cancel_and_cleanup(_patch_export_backend):
    from mcp_server.tools.export import export_cancel, export_cleanup

    backend = _patch_export_backend(_FakeExportBackend())
    assert export_cancel()["message"] == "Export cancelled"
    assert backend.cancelled is True
    assert export_cleanup()["success"] is True
    assert backend.cleaned is True


def test_export_merged_passes_params(_patch_export_backend):
    from mcp_server.tools.export import export_merged

    backend = _FakeExportBackend()
    backend.export_merged_model = lambda **kw: (True, "ok", "/out/merged")
    _patch_export_backend(backend)
    result = export_merged(save_directory="/out", format_type="4-bit (FP4)")
    assert result["success"] is True
    assert result["output_path"] == "/out/merged"


# ── recipe tools (mocked manager) ─────────────────────────────────────


class _FakeJobManager:
    def __init__(self):
        self.started = None
        self.cancelled_id = None

    def start(self, *, recipe, run, internal_api_key_id=None):
        self.started = {"recipe": recipe, "run": run}
        return "recipe_job_123"

    def get_status(self, job_id):
        if job_id != "recipe_job_123":
            return None
        return {"job_id": job_id, "status": "running", "rows": 10}

    def get_current_status(self):
        return {"job_id": "recipe_job_123", "status": "running"}

    def cancel(self, job_id):
        self.cancelled_id = job_id
        return True


@pytest.fixture
def _patch_job_manager(monkeypatch):
    def _install(mgr):
        import core.data_recipe.jobs.manager as mod

        monkeypatch.setattr(mod, "get_job_manager", lambda: mgr)
        return mgr

    return _install


def test_recipe_start_requires_columns(_patch_job_manager):
    from mcp_server.tools.recipe import recipe_start

    _patch_job_manager(_FakeJobManager())
    result = recipe_start(recipe={})
    assert result["success"] is False
    assert "columns" in result["error"]


def test_recipe_start_spawns_job(_patch_job_manager):
    from mcp_server.tools.recipe import recipe_start

    mgr = _patch_job_manager(_FakeJobManager())
    result = recipe_start(recipe={"columns": [{"name": "x"}]}, run={"execution_type": "preview"})
    assert result["success"] is True
    assert result["job_id"] == "recipe_job_123"
    assert mgr.started["run"]["execution_type"] == "preview"


def test_recipe_start_rejects_bad_execution_type(_patch_job_manager):
    from mcp_server.tools.recipe import recipe_start

    _patch_job_manager(_FakeJobManager())
    result = recipe_start(recipe={"columns": [{}]}, run={"execution_type": "weird"})
    assert result["success"] is False


def test_recipe_status_by_id(_patch_job_manager):
    from mcp_server.tools.recipe import recipe_status

    _patch_job_manager(_FakeJobManager())
    assert recipe_status("recipe_job_123")["status"] == "running"
    assert recipe_status("other")["idle"] is True


def test_recipe_status_current(_patch_job_manager):
    from mcp_server.tools.recipe import recipe_status

    _patch_job_manager(_FakeJobManager())
    assert recipe_status()["status"] == "running"


def test_recipe_cancel(_patch_job_manager):
    from mcp_server.tools.recipe import recipe_cancel

    mgr = _patch_job_manager(_FakeJobManager())
    result = recipe_cancel("recipe_job_123")
    assert result["success"] is True
    assert mgr.cancelled_id == "recipe_job_123"


def test_recipe_validate_requires_columns():
    from mcp_server.tools.recipe import recipe_validate

    assert recipe_validate({})["valid"] is False


# ── integration: FastMCP in-memory client ─────────────────────────────


def test_build_server_registers_all_tools():
    from fastmcp import Client

    from mcp_server import build_server

    server = build_server()
    tools = asyncio.run(_list_tools(server))
    assert len(tools) == 23
    names = {t.name for t in tools}
    for expected in [
        "models_list",
        "data_list_local",
        "train_start",
        "export_gguf",
        "recipe_start",
    ]:
        assert expected in names


def test_build_server_gated_subset():
    from fastmcp import Client

    from mcp_server import build_server

    server = build_server(enabled=["models", "data"])
    tools = asyncio.run(_list_tools(server))
    names = {t.name for t in tools}
    assert "models_list" in names
    assert "data_register" in names
    assert "train_start" not in names
    assert "export_gguf" not in names


def test_read_only_tools_carry_annotation():
    from fastmcp import Client

    from mcp_server import build_server

    server = build_server()
    tools = {t.name: t for t in asyncio.run(_list_tools(server))}
    assert tools["models_list"].annotations.readOnlyHint is True
    assert tools["data_list_local"].annotations.readOnlyHint is True
    # stateful tools are NOT marked read-only (unset => None, not True)
    assert tools["train_start"].annotations.readOnlyHint is not True


def test_client_can_call_tool_end_to_end():
    from fastmcp import Client

    from mcp_server import build_server

    server = build_server(enabled=["models"])
    result = asyncio.run(_call_tool(server, "models_list", {}))
    assert result["count"] > 0


def test_build_server_rejects_unknown_group():
    from mcp_server import build_server

    with pytest.raises(ValueError, match="Unknown feature group"):
        build_server(enabled=["bogus"])


async def _list_tools(server):
    from fastmcp import Client

    async with Client(server) as client:
        return await client.list_tools()


async def _call_tool(server, name, arguments):
    from fastmcp import Client

    async with Client(server) as client:
        result = await client.call_tool(name, arguments)
        return result.data
