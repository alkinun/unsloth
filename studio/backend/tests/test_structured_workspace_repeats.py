# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Structured and textual batches must not replay a successful workspace edit."""

import json

import pytest

from .test_llama_cpp_tool_loop import (
    _backend_and_payloads,
    _done,
    _run_tool_loop,
    _sse,
    _tool_call_sse,
)


def _run_batch(
    monkeypatch,
    batch,
    execute,
    *,
    structured = True,
):
    if structured:
        stream = [
            _tool_call_sse(name, arguments, f"call-{i}", i)
            for i, (name, arguments) in enumerate(batch)
        ]
    else:
        stream = [
            _sse(
                {
                    "content": "".join(
                        "<tool_call>"
                        + json.dumps({"name": name, "arguments": arguments})
                        + "</tool_call>"
                        for name, arguments in batch
                    )
                }
            )
        ]
    backend, payloads = _backend_and_payloads(
        monkeypatch, [stream + [_done()], [_sse({"content": "Done."}), _done()]]
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *a, **k: 100)
    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Edit and verify"}],
        [
            {"type": "function", "function": {"name": name}}
            for name in ("terminal", "edit_file", "web_search")
        ],
        max_tool_iterations = 3,
    )
    return events, payloads


@pytest.mark.parametrize("structured", [False, True])
def test_repeating_batch_applies_an_append_once(monkeypatch, tmp_path, structured):
    target = tmp_path / "log.txt"
    target.write_text("")
    read = ("terminal", {"command": "cat log.txt"})
    append = ("terminal", {"command": "echo x >> log.txt"})
    seen = []

    def execute(name, arguments, **kwargs):
        seen.append(arguments["command"])
        if arguments["command"] == append[1]["command"]:
            with target.open("a") as output:
                output.write("x\n")
        return target.read_text()

    _, payloads = _run_batch(
        monkeypatch, [read, append, read, append], execute, structured = structured
    )
    assert target.read_text() == "x\n"
    assert seen == [read[1]["command"], append[1]["command"], read[1]["command"]]
    messages = payloads[-1]["messages"]
    calls = [tc for m in messages for tc in m.get("tool_calls", [])]
    results = [m for m in messages if m["role"] == "tool"]
    assert {c["id"] for c in calls} == {r["tool_call_id"] for r in results}


def test_structured_batch_keeps_adjacent_duplicates_on_the_controller_guard(monkeypatch):
    read = ("terminal", {"command": "cat log.txt"})
    seen = []
    _, payloads = _run_batch(monkeypatch, [read] * 3, lambda *a, **k: seen.append(a) or "x")
    assert len(seen) == 1
    assert not payloads[-1].get("tools")


def test_structured_batch_keeps_more_than_eight_distinct_calls(monkeypatch):
    seen = []
    batch = [("terminal", {"command": f"cat file-{i}.txt"}) for i in range(10)]
    _run_batch(monkeypatch, batch, lambda *a, **k: seen.append(a) or "x")
    assert len(seen) == 10


def test_structured_batch_verifies_each_independent_edit(monkeypatch):
    read = ("terminal", {"command": "pytest -q"})
    edit_a = ("edit_file", {"path": "a.py", "edits": []})
    edit_b = ("edit_file", {"path": "b.py", "edits": []})
    seen = []
    batch = [read, edit_a, read, edit_b, read]
    _run_batch(monkeypatch, batch, lambda name, args, **k: seen.append((name, args)) or "OK")
    assert seen == batch


def test_filtered_structured_repeat_closes_its_provisional_card(monkeypatch):
    read = ("terminal", {"command": "cat log.txt"})
    append = ("terminal", {"command": "echo x >> log.txt # " + "explanation " * 30})
    seen = []
    events, payloads = _run_batch(
        monkeypatch,
        [read, append, read, append],
        lambda name, args, **k: seen.append((name, args)) or "OK",
    )
    assert seen == [read, append, read]
    assert any(e.get("type") == "tool_start" and e.get("tool_call_id") == "call-3" for e in events)
    ends = [e for e in events if e.get("type") == "tool_end" and e.get("tool_call_id") == "call-3"]
    assert len(ends) == 1 and ends[0]["result"] == ""
    assert not any(
        tc["id"] == "call-3" for m in payloads[-1]["messages"] for tc in m.get("tool_calls", [])
    )
