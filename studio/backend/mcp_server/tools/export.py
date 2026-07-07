# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP tools for Studio model export.

Export operations (``export_load_checkpoint``, ``export_merged`` /
``export_gguf`` / ``export_lora``) are **synchronous** and block until the
operation finishes -- they may take minutes (merged) to hours (multi-quant
GGUF). They run on the Studio export subprocess, exactly like the Studio UI.
Use ``export_status`` for the loaded-checkpoint state and ``export_cancel`` to
abort an in-flight export.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from fastmcp import FastMCP

from loggers import get_logger
from mcp_server.auth import resolve_hf_token

logger = get_logger(__name__)

GROUP = "export"

_READ_ONLY = {"readOnlyHint": True}
_STATEFUL = {"destructiveHint": False}


def export_list_checkpoints(outputs_dir: Optional[str] = None) -> dict[str, Any]:
    """List training checkpoints available for export.

    Scans Studio's outputs root (or ``outputs_dir``) for runs/checkpoints.
    Pass one of the returned checkpoint paths to ``export_load_checkpoint``.
    """
    from core.export import get_export_backend

    backend = get_export_backend()
    scanned = backend.scan_checkpoints(outputs_dir=outputs_dir)
    runs: list[dict[str, Any]] = []
    for model_name, checkpoints, metadata in scanned:
        runs.append(
            {
                "model_name": model_name,
                "checkpoints": [
                    {"name": name, "path": path, "loss": loss} for name, path, loss in checkpoints
                ],
                "metadata": metadata,
            }
        )
    return {"outputs_dir": outputs_dir, "runs": runs}


def export_load_checkpoint(
    checkpoint_path: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    trust_remote_code: bool = False,
    hf_token: Optional[str] = None,
) -> dict[str, Any]:
    """Load a checkpoint into the export backend (must precede ``export_*``).

    Returns once the checkpoint is resident in the export subprocess. After a
    successful load, call one of ``export_merged`` / ``export_gguf`` /
    ``export_lora``.
    """
    from core.export import get_export_backend

    backend = get_export_backend()
    success, message = backend.load_checkpoint(
        checkpoint_path=checkpoint_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
        hf_token=resolve_hf_token(hf_token),
        subject="mcp",
    )
    return {"success": success, "message": message, "checkpoint": checkpoint_path}


def export_merged(
    save_directory: str,
    format_type: str = "16-bit (FP16)",
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    private: bool = False,
    compressed_method: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> dict[str, Any]:
    """Export the loaded checkpoint as a merged model.

    ``format_type`` is one of ``16-bit (FP16)``, ``4-bit (FP4)``,
    ``FP8 (compressed-tensors)``, ``NVFP4 (compressed-tensors)``. Blocks until
    the merge writes to ``save_directory`` (and optionally pushes to the Hub).
    """
    from core.export import get_export_backend

    backend = get_export_backend()
    success, message, output_path = backend.export_merged_model(
        save_directory=save_directory,
        format_type=format_type,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        hf_token=resolve_hf_token(hf_token),
        private=private,
        compressed_method=compressed_method,
    )
    return _export_result("merged", success, message, output_path)


def export_gguf(
    save_directory: str,
    quantization_method: Union[str, List[str]] = "Q4_K_M",
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    imatrix_file: Optional[str] = None,
) -> dict[str, Any]:
    """Export the loaded checkpoint in GGUF format for llama.cpp / Ollama.

    ``quantization_method`` is a llama.cpp quant (e.g. ``Q4_K_M``, ``Q8_0``,
    ``F16``) or a list of quants to produce in one pass off a single merge.
    This is the slowest export (minutes to hours for large models).
    """
    from core.export import get_export_backend

    backend = get_export_backend()
    success, message, output_path = backend.export_gguf(
        save_directory=save_directory,
        quantization_method=quantization_method,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        hf_token=resolve_hf_token(hf_token),
        imatrix_file=imatrix_file,
    )
    return _export_result("gguf", success, message, output_path)


def export_lora(
    save_directory: str,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    private: bool = False,
    gguf: bool = False,
    gguf_outtype: str = "q8_0",
    hf_token: Optional[str] = None,
) -> dict[str, Any]:
    """Export the loaded checkpoint as a LoRA adapter only.

    With ``gguf=True``, also writes a GGUF LoRA file (``gguf_outtype`` one of
    ``q8_0``, ``f16``, ``bf16``, ``f32``). Much smaller/faster than a full merge.
    """
    from core.export import get_export_backend

    backend = get_export_backend()
    success, message, output_path = backend.export_lora_adapter(
        save_directory=save_directory,
        push_to_hub=push_to_hub,
        repo_id=repo_id,
        hf_token=resolve_hf_token(hf_token),
        private=private,
        gguf=gguf,
        gguf_outtype=gguf_outtype,
    )
    return _export_result("lora", success, message, output_path)


def export_status() -> dict[str, Any]:
    """Get export-backend state: loaded checkpoint, model type, last operation.

    Use this after ``export_load_checkpoint`` to confirm what is resident
    before choosing an ``export_*`` format, and to inspect the last op result.
    """
    from core.export import get_export_backend

    backend = get_export_backend()
    last_op = backend.get_last_op() or {}
    return {
        "current_checkpoint": backend.current_checkpoint,
        "is_vision": bool(getattr(backend, "is_vision", False)),
        "is_peft": bool(getattr(backend, "is_peft", False)),
        "is_export_active": bool(backend.is_export_active()),
        "active_op_kind": backend.get_active_op_kind(),
        "last_op": {
            "kind": last_op.get("kind"),
            "status": last_op.get("status"),
            "output_path": last_op.get("output_path"),
            "error": last_op.get("error"),
        },
    }


def export_cancel() -> dict[str, Any]:
    """Cancel any in-flight export (terminates the export subprocess)."""
    from core.export import get_export_backend

    backend = get_export_backend()
    cancelled = backend.cancel_export()
    return {
        "success": True,
        "message": "Export cancelled" if cancelled else "No active export to cancel",
    }


def export_cleanup() -> dict[str, Any]:
    """Release the loaded checkpoint / free export GPU memory."""
    from core.export import get_export_backend

    backend = get_export_backend()
    success = backend.cleanup_memory()
    return {
        "success": success,
        "message": "Memory cleanup completed" if success else "Cleanup failed",
    }


def _export_result(
    kind: str, success: bool, message: str, output_path: Optional[str]
) -> dict[str, Any]:
    return {
        "success": success,
        "kind": kind,
        "message": message,
        "output_path": output_path,
    }


def register(mcp: FastMCP) -> list[str]:
    """Register the export tools onto ``mcp``; return the tool names added."""
    names: list[str] = []
    mcp.tool(export_list_checkpoints, annotations=_READ_ONLY)
    names.append("export_list_checkpoints")
    mcp.tool(export_status, annotations=_READ_ONLY)
    names.append("export_status")
    for fn in (
        export_load_checkpoint,
        export_merged,
        export_gguf,
        export_lora,
        export_cancel,
        export_cleanup,
    ):
        mcp.tool(fn, annotations=_STATEFUL)
        names.append(fn.__name__)
    return names
