"""Wraps files.py / processes.py / archives.py as `contracts.Tool` instances
with Anthropic-style JSON schemas, so Sara's `solver/registry.py` can drop
`get_tools()` straight into `tools=[...]` on a `messages.create` call.

Tool NAMES AND ARGUMENT SCHEMAS below are a de facto contract with the
solver even though `contracts.py` doesn't pin per-tool shapes — see this
module's PR description for the list Sara should wire up.

`contracts.py` doesn't exist in this branch yet (Nandh's Gate 1 hasn't
landed), so the import below falls back to a local stub that mirrors it
verbatim. Swap to the bare `from contracts import ...` once that file exists
and delete `_contracts_stub.py` — every other module in this package is
contracts-agnostic and needs no change.
"""
from __future__ import annotations

import time

try:
    from contracts import TaskContext, ToolRequest, ToolResult
except ImportError:
    from tools.runtime._contracts_stub import TaskContext, ToolRequest, ToolResult

from tools.runtime import archives, files, processes, python_exec
from tools.runtime.errors import INVALID_ARGUMENT, PROCESS_FAILED, TIMEOUT, RuntimeToolError
from tools.runtime.processes import ProcessResult

ANSWER_MARKER = "ANSWER:"
_MAX_LISTED_MEMBERS = 200


# ---------------------------------------------------------------- base

class _BaseTool:
    name: str
    description: str
    input_schema: dict

    def execute(self, request: ToolRequest, task: TaskContext) -> ToolResult:
        start = time.monotonic()
        try:
            output, exact_value = self._run(request, task)
            return ToolResult(ok=True, output=output,
                              elapsed_ms=_elapsed_ms(start), exact_value=exact_value)
        except RuntimeToolError as e:
            return ToolResult(ok=False, output=e.message, error_code=e.code,
                              elapsed_ms=_elapsed_ms(start))
        except Exception as e:  # noqa: BLE001 — never leak a raw traceback to the model
            return ToolResult(ok=False, output=f"unexpected {type(e).__name__}: {e}",
                              error_code=PROCESS_FAILED, elapsed_ms=_elapsed_ms(start))

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
               "input_schema": self.input_schema}

    def _run(self, request: ToolRequest, task: TaskContext) -> tuple[str, str | None]:
        raise NotImplementedError


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _require_str(arguments: dict, key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeToolError(INVALID_ARGUMENT, f"{key!r} must be a non-empty string")
    return value


def _extract_answer_marker(stdout: str) -> str | None:
    """If the process printed a line starting with `ANSWER:`, return the
    text after the LAST such line, stripped. This is the structural fix the
    README asks for: an exact token that never round-trips through the
    model's own retyping, because it flows program stdout -> ToolResult
    .exact_value -> CandidateAnswer.value untouched.
    """
    value = None
    for line in stdout.splitlines():
        if line.startswith(ANSWER_MARKER):
            value = line[len(ANSWER_MARKER):].strip()
    return value


def _format_process_result(result: ProcessResult) -> str:
    parts = [
        f"exit_code={result.exit_code} timed_out={result.timed_out} "
        f"elapsed_ms={result.elapsed_ms}",
        f"--- stdout ---\n{result.stdout}",
        f"--- stderr ---\n{result.stderr}",
    ]
    return "\n".join(parts)


def _run_and_report(result: ProcessResult) -> tuple[str, str | None]:
    output = _format_process_result(result)
    if result.timed_out:
        raise RuntimeToolError(TIMEOUT, output)
    if result.exit_code != 0:
        raise RuntimeToolError(PROCESS_FAILED, output)
    return output, _extract_answer_marker(result.stdout)


# ---------------------------------------------------------------- files

class ListFilesTool(_BaseTool):
    name = "list_files"
    description = ("List the files and subdirectories directly inside a path "
                   "in the task's working directory. Not recursive.")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Path relative to the task workdir. Omit for the root."},
        },
    }

    def _run(self, request: ToolRequest, task: TaskContext) -> tuple[str, str | None]:
        sub_path = request.arguments.get("path", ".")
        entries = files.list_dir(task.workdir, sub_path)
        if not entries:
            return "(empty directory)", None
        lines = [f"{'d' if e.is_dir else 'f'} {e.size:>10} {e.path}" for e in entries]
        return "\n".join(lines), None


class ReadFileTool(_BaseTool):
    name = "read_file"
    description = (
        "Read a file from the task workdir. Defaults to the whole file as text "
        "(bounded, with a truncation marker if it doesn't fit). Pass start_line"
        "/end_line for a line range, or offset/length with encoding=\"binary\" "
        "for a byte range of a binary file returned as base64.")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the task workdir."},
            "encoding": {"type": "string", "enum": ["text", "binary"], "default": "text"},
            "start_line": {"type": "integer", "description": "1-indexed, inclusive."},
            "end_line": {"type": "integer", "description": "1-indexed, inclusive."},
            "offset": {"type": "integer", "description": "Byte offset (binary range read)."},
            "length": {"type": "integer", "description": "Byte length (binary range read)."},
        },
        "required": ["path"],
    }

    def _run(self, request: ToolRequest, task: TaskContext) -> tuple[str, str | None]:
        arguments = request.arguments
        path = _require_str(arguments, "path")
        if "offset" in arguments or "length" in arguments:
            offset = int(arguments.get("offset", 0))
            length = int(arguments.get("length", files.DEFAULT_MAX_READ_BYTES))
            text, _ = files.read_byte_range(task.workdir, path, offset, length)
            return text, None
        if "start_line" in arguments or "end_line" in arguments:
            start = int(arguments.get("start_line", 1))
            end = int(arguments.get("end_line", start))
            text, _ = files.read_line_range(task.workdir, path, start, end)
            return text, None
        if arguments.get("encoding") == "binary":
            text, _ = files.read_bytes(task.workdir, path)
            return text, None
        text, _ = files.read_text(task.workdir, path)
        return text, None


class WriteScratchFileTool(_BaseTool):
    name = "write_scratch_file"
    description = "Write a text file into the task workdir (creating parent dirs as needed)."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the task workdir."},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def _run(self, request: ToolRequest, task: TaskContext) -> tuple[str, str | None]:
        arguments = request.arguments
        path = _require_str(arguments, "path")
        content = arguments.get("content")
        if not isinstance(content, str):
            raise RuntimeToolError(INVALID_ARGUMENT, "'content' must be a string")
        n = files.write_scratch(task.workdir, path, content)
        return f"wrote {n} bytes to {path}", None


# ---------------------------------------------------------------- execution

class RunPythonTool(_BaseTool):
    name = "run_python"
    description = (
        "Run a Python code snippet in a subprocess, cwd set to the task workdir, "
        "bounded by this call's timeout. Print a line starting with 'ANSWER:' to "
        "have that exact text (after the marker) returned as this tool's "
        "exact_value, bypassing the need to retype it later.")
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to run."},
            "args": {"type": "array", "items": {"type": "string"},
                    "description": "Extra argv passed to the script (sys.argv[1:])."},
        },
        "required": ["code"],
    }

    def _run(self, request: ToolRequest, task: TaskContext) -> tuple[str, str | None]:
        code = _require_str(request.arguments, "code")
        args = request.arguments.get("args")
        result = python_exec.run_python_code(
            task.workdir, code, timeout_seconds=request.timeout_seconds, args=args)
        return _run_and_report(result)


class RunProcessTool(_BaseTool):
    name = "run_process"
    description = (
        "Run an arbitrary command (argv list, no shell) in the task workdir, "
        "bounded by this call's timeout. Same 'ANSWER:' stdout convention as "
        "run_python.")
    input_schema = {
        "type": "object",
        "properties": {
            "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                    "description": "Command and arguments, e.g. [\"ls\", \"-la\"]. No shell syntax."},
            "cpu_heavy": {"type": "boolean", "default": True,
                         "description": "Count this against the 2-concurrent CPU-heavy job cap."},
        },
        "required": ["argv"],
    }

    def _run(self, request: ToolRequest, task: TaskContext) -> tuple[str, str | None]:
        argv = request.arguments.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise RuntimeToolError(INVALID_ARGUMENT, "'argv' must be a non-empty array of strings")
        cpu_heavy = bool(request.arguments.get("cpu_heavy", True))
        result = processes.run_process(
            task.workdir, argv, timeout_seconds=request.timeout_seconds, cpu_heavy=cpu_heavy)
        return _run_and_report(result)


# ---------------------------------------------------------------- archives

class InspectArchiveTool(_BaseTool):
    name = "inspect_archive"
    description = "Inspect a ZIP or TAR archive's members and total size without extracting."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Archive path relative to the task workdir."},
        },
        "required": ["path"],
    }

    def _run(self, request: ToolRequest, task: TaskContext) -> tuple[str, str | None]:
        path = _require_str(request.arguments, "path")
        info = archives.inspect(task.workdir, path)
        lines = [f"format={info.format} member_count={info.member_count} "
                f"total_uncompressed_size={info.total_uncompressed_size}"]
        shown = info.members[:_MAX_LISTED_MEMBERS]
        lines += [f"{'d' if m.is_dir else 'f'} {m.size:>10} {m.name}" for m in shown]
        if len(info.members) > len(shown):
            lines.append(f"... and {len(info.members) - len(shown)} more members")
        return "\n".join(lines), None


class ExtractArchiveTool(_BaseTool):
    name = "extract_archive"
    description = (
        "Extract a ZIP or TAR archive into a directory inside the task workdir. "
        "Set recursive=true to also extract archives found inside it, up to a "
        "depth limit, under one cumulative size/member budget.")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Archive path relative to the task workdir."},
            "dest": {"type": "string", "description": "Destination directory relative to the task workdir."},
            "recursive": {"type": "boolean", "default": False},
        },
        "required": ["path", "dest"],
    }

    def _run(self, request: ToolRequest, task: TaskContext) -> tuple[str, str | None]:
        arguments = request.arguments
        path = _require_str(arguments, "path")
        dest = _require_str(arguments, "dest")
        if arguments.get("recursive"):
            result = archives.extract_recursive(task.workdir, path, dest)
        else:
            result = archives.extract(task.workdir, path, dest)
        lines = [f"extracted {result.extracted_count} members "
                f"({result.extracted_bytes} bytes) into {result.dest}"]
        if result.nested_archives:
            lines.append("nested archives found (pass recursive=true to descend into them): "
                        + ", ".join(result.nested_archives))
        return "\n".join(lines), None


# ---------------------------------------------------------------- registry

def get_tools() -> list[_BaseTool]:
    """Every runtime tool, ready to hand to Sara's registry."""
    return [
        ListFilesTool(), ReadFileTool(), WriteScratchFileTool(),
        RunPythonTool(), RunProcessTool(),
        InspectArchiveTool(), ExtractArchiveTool(),
    ]


def get_schemas() -> list[dict]:
    """JSON schemas ready for the Anthropic SDK's `tools=[...]` parameter."""
    return [t.schema() for t in get_tools()]
