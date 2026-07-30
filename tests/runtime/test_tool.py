from __future__ import annotations

import base64
import time
import zipfile
from pathlib import Path

from tools.runtime.errors import INVALID_ARGUMENT, PATH_BLOCKED, PROCESS_FAILED, TIMEOUT
from tools.runtime.tool import (
    ExtractArchiveTool, InspectArchiveTool, ListFilesTool, ReadFileTool,
    RunProcessTool, RunPythonTool, TaskContext, ToolRequest, WriteScratchFileTool,
    _BaseTool, _MAX_MODEL_READ_BYTES, _MAX_MODEL_STREAM_CHARS, get_schemas,
    get_tools,
)


def _task(workdir: Path) -> TaskContext:
    return TaskContext(
        task_id="T-1", category="Cryptic", points=100, prompt="solve it",
        answer_format="exact", workdir=workdir, files=(),
        deadline_monotonic=time.monotonic() + 60, metadata={})


def _req(name: str, arguments: dict, timeout_seconds: float = 10) -> ToolRequest:
    return ToolRequest(name=name, arguments=arguments, timeout_seconds=timeout_seconds)


def test_get_tools_and_schemas_agree_on_names() -> None:
    tools = get_tools()
    schemas = get_schemas()
    assert [t.name for t in tools] == [s["name"] for s in schemas]
    assert len(tools) == len(set(t.name for t in tools))  # unique names


def test_list_files_tool(workdir: Path) -> None:
    (workdir / "a.txt").write_text("hi")
    result = ListFilesTool().execute(_req("list_files", {}), _task(workdir))
    assert result.ok
    assert "a.txt" in result.output


def test_read_file_tool_happy_path(workdir: Path) -> None:
    (workdir / "a.txt").write_text("hello")
    result = ReadFileTool().execute(_req("read_file", {"path": "a.txt"}), _task(workdir))
    assert result.ok
    assert result.output == "hello"


def test_read_file_tool_caps_whole_text_before_model_context(workdir: Path) -> None:
    (workdir / "large.txt").write_text("x" * (_MAX_MODEL_READ_BYTES * 4))

    result = ReadFileTool().execute(
        _req("read_file", {"path": "large.txt"}),
        _task(workdir),
    )

    assert result.ok
    assert result.output.startswith("x" * _MAX_MODEL_READ_BYTES)
    assert "read_file capped at 12000 bytes" in result.output
    assert len(result.output) < _MAX_MODEL_READ_BYTES + 300


def test_read_file_tool_preserves_small_targeted_line_range(workdir: Path) -> None:
    lines = [f"line-{index}: {'x' * 80}" for index in range(1, 2_001)]
    (workdir / "large.txt").write_text("\n".join(lines) + "\n")

    result = ReadFileTool().execute(
        _req(
            "read_file",
            {"path": "large.txt", "start_line": 1_337, "end_line": 1_338},
        ),
        _task(workdir),
    )

    assert result.ok
    assert result.output == f"{lines[1336]}\n{lines[1337]}\n"
    assert "truncated" not in result.output


def test_read_file_tool_caps_large_line_range_with_actionable_marker(
    workdir: Path,
) -> None:
    (workdir / "lines.txt").write_text(
        "\n".join(f"line-{index}: {'y' * 100}" for index in range(1, 1_001))
        + "\n"
    )

    result = ReadFileTool().execute(
        _req(
            "read_file",
            {"path": "lines.txt", "start_line": 1, "end_line": 1_000},
        ),
        _task(workdir),
    )

    assert result.ok
    assert "line-1:" in result.output
    assert "line-1000:" not in result.output
    assert "read_file capped at 12000 bytes" in result.output
    assert "start_line/end_line" in result.output


def test_read_file_tool_caps_binary_before_base64_expansion(workdir: Path) -> None:
    payload = bytes(range(256)) * 100
    (workdir / "large.bin").write_bytes(payload)

    result = ReadFileTool().execute(
        _req("read_file", {"path": "large.bin", "encoding": "binary"}),
        _task(workdir),
    )

    assert result.ok
    encoded, marker = result.output.split("\n...", maxsplit=1)
    assert base64.b64decode(encoded) == payload[:_MAX_MODEL_READ_BYTES]
    assert "read_file capped at 12000 bytes" in marker


def test_read_file_tool_preserves_targeted_byte_range(workdir: Path) -> None:
    payload = bytes(range(256)) * 100
    (workdir / "large.bin").write_bytes(payload)

    result = ReadFileTool().execute(
        _req(
            "read_file",
            {"path": "large.bin", "offset": 15_000, "length": 32},
        ),
        _task(workdir),
    )

    assert result.ok
    assert base64.b64decode(result.output) == payload[15_000:15_032]
    assert "truncated" not in result.output


def test_read_file_tool_blocks_traversal(workdir: Path) -> None:
    result = ReadFileTool().execute(
        _req("read_file", {"path": "../../secret"}), _task(workdir))
    assert not result.ok
    assert result.error_code == PATH_BLOCKED


def test_write_scratch_file_tool(workdir: Path) -> None:
    result = WriteScratchFileTool().execute(
        _req("write_scratch_file", {"path": "out.txt", "content": "data"}), _task(workdir))
    assert result.ok
    assert (workdir / "out.txt").read_text() == "data"


def test_run_python_tool_answer_marker_becomes_exact_value(workdir: Path) -> None:
    code = "print('some reasoning'); print('ANSWER: GHOST-6SAS2HPHXQ5V')"
    result = RunPythonTool().execute(_req("run_python", {"code": code}), _task(workdir))
    assert result.ok
    assert result.exact_value == "GHOST-6SAS2HPHXQ5V"


def test_run_python_exact_value_survives_verbose_output(workdir: Path) -> None:
    """A chatty program must not lose its answer or flood the model context.

    Printing a large table before the result is exactly what the data-wrangling
    categories invite. The answer is extracted from the full capture, so it has
    to survive, while the text handed to the model stays bounded — an unclipped
    200 KB stream is roughly 50k tokens, enough on its own to exhaust the
    solver's cumulative token budget and get the tile retried from scratch.
    """
    code = (
        "for _ in range(20000): print('x' * 20)\n"
        "print('ANSWER: GHOST-6SAS2HPHXQ5V')\n"
    )
    result = RunPythonTool().execute(
        _req("run_python", {"code": code}, timeout_seconds=60), _task(workdir))

    assert result.ok
    assert result.exact_value == "GHOST-6SAS2HPHXQ5V"
    assert "ANSWER: GHOST-6SAS2HPHXQ5V" in result.output
    assert "clipped:" in result.output
    assert len(result.output) < 4 * _MAX_MODEL_STREAM_CHARS


def test_small_process_output_is_passed_through_unclipped(workdir: Path) -> None:
    code = "print('concise'); print('ANSWER: 42')"
    result = RunPythonTool().execute(_req("run_python", {"code": code}), _task(workdir))

    assert result.ok
    assert result.exact_value == "42"
    assert "clipped:" not in result.output
    assert "truncated" not in result.output


def test_run_python_tool_timeout_reported_as_typed_error(workdir: Path) -> None:
    code = "import time; time.sleep(60)"
    result = RunPythonTool().execute(_req("run_python", {"code": code}, timeout_seconds=0.5),
                                     _task(workdir))
    assert not result.ok
    assert result.error_code == TIMEOUT


def test_run_python_tool_nonzero_exit_is_process_failed(workdir: Path) -> None:
    code = "import sys; sys.exit(1)"
    result = RunPythonTool().execute(_req("run_python", {"code": code}), _task(workdir))
    assert not result.ok
    assert result.error_code == PROCESS_FAILED


def test_run_process_tool_rejects_bad_argv(workdir: Path) -> None:
    result = RunProcessTool().execute(_req("run_process", {"argv": "echo hi"}), _task(workdir))
    assert not result.ok
    assert result.error_code == INVALID_ARGUMENT


def test_inspect_and_extract_archive_tools(workdir: Path) -> None:
    with zipfile.ZipFile(workdir / "a.zip", "w") as zf:
        zf.writestr("one.txt", "hello")

    inspected = InspectArchiveTool().execute(
        _req("inspect_archive", {"path": "a.zip"}), _task(workdir))
    assert inspected.ok
    assert "member_count=1" in inspected.output

    extracted = ExtractArchiveTool().execute(
        _req("extract_archive", {"path": "a.zip", "dest": "out"}), _task(workdir))
    assert extracted.ok
    assert (workdir / "out" / "one.txt").read_text() == "hello"


def test_execute_never_leaks_raw_exception(workdir: Path) -> None:
    class ExplodingTool(_BaseTool):
        name = "exploding_tool"
        description = "test double"
        input_schema = {"type": "object", "properties": {}}

        def _run(self, request, task):
            raise ValueError("boom")

    result = ExplodingTool().execute(_req("exploding_tool", {}), _task(workdir))
    assert not result.ok
    assert result.error_code == PROCESS_FAILED
    assert "boom" in result.output
