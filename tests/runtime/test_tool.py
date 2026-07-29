from __future__ import annotations

import time
import zipfile
from pathlib import Path

from tools.runtime.errors import INVALID_ARGUMENT, PATH_BLOCKED, PROCESS_FAILED, TIMEOUT
from tools.runtime.tool import (
    ExtractArchiveTool, InspectArchiveTool, ListFilesTool, ReadFileTool,
    RunProcessTool, RunPythonTool, TaskContext, ToolRequest, WriteScratchFileTool,
    _BaseTool, get_schemas, get_tools,
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
