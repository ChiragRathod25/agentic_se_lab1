"""Task 4: Tool layer implementation, strict schema validation and containment.
Provides ToolRegistry implementing ToolRegistryBase from tool_contract.py.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import jsonschema

from tool_contract import (
    DEFAULT_TIMEOUT_S,
    MAX_LIST_ENTRIES,
    MAX_OUTPUT_CHARS,
    MAX_READ_LINES,
    MAX_SEARCH_RESULTS,
    ToolRegistryBase,
    ToolResult,
)


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the workspace directory matching an optional glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path to list (default: '.').",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional glob pattern to filter entries (e.g. '*.py' or '**/*.py').",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents with 1-based line numbers, optionally within a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path within the workspace.",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional starting line number (1-based, inclusive).",
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional ending line number (1-based, inclusive).",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    "search_code": {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search workspace files using a regular expression pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional glob filter for filenames (e.g. '**/*.py').",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                        "description": f"Maximum matches to return (up to {MAX_SEARCH_RESULTS}).",
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    "run_tests": {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest test suite on the workspace and return a structured summary of passed/failed tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Optional test file or node id (e.g. 'tests/test_settle.py' or 'tests/test_settle.py::test_name').",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    "run_cli": {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": "Run the tinyledger CLI module (python -m tinyledger <args>) and return exit code, stdout, and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of command line arguments to pass to tinyledger.",
                    },
                },
                "required": ["args"],
                "additionalProperties": False,
            },
        },
    },
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file under src/ by replacing a unique exact string with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path (must be strictly inside src/).",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact existing text to be replaced (must match uniquely in the file).",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old_str", "new_str"],
                "additionalProperties": False,
            },
        },
    },
}


class ToolRegistry(ToolRegistryBase):
    """Secure registry for agent tools with schema validation and containment."""

    def __init__(self, workspace: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        super().__init__(workspace=workspace, timeout_s=timeout_s)
        self.workspace = self.workspace.resolve()

    def schemas(self) -> list[dict]:
        """Return OpenAI function tool definitions."""
        return list(TOOL_SCHEMAS.values())

    def execute(self, name: str, arguments: str) -> ToolResult:
        """Validate and execute one tool call safely. Never raises."""
        try:
            if name not in TOOL_SCHEMAS:
                return ToolResult(ok=False, error=f"Unknown tool: '{name}'. Available tools: {list(TOOL_SCHEMAS.keys())}")

            # Parse JSON arguments
            try:
                args = json.loads(arguments) if arguments else {}
            except Exception as exc:
                return ToolResult(ok=False, error=f"Invalid JSON arguments: {exc}")

            if not isinstance(args, dict):
                return ToolResult(ok=False, error="Arguments must be a JSON object.")

            # Validate against tool parameters schema
            schema = TOOL_SCHEMAS[name]["function"]["parameters"]
            try:
                jsonschema.validate(instance=args, schema=schema)
            except jsonschema.ValidationError as val_err:
                return ToolResult(ok=False, error=f"Schema validation error: {val_err.message}")

            # Dispatch to handler
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return ToolResult(ok=False, error=f"Tool handler for '{name}' is not implemented.")

            result = handler(**args)
            return self._bound_result(result)

        except Exception as exc:
            # Defensive guard: execute must never crash the caller
            return ToolResult(ok=False, error=f"Unexpected error executing {name}: {exc}")

    # Containment helpers
    def _resolve_safe_path(self, rel_path: str, must_exist: bool = True) -> tuple[Path | None, str | None]:
        """Resolve a relative path, enforcing that it stays strictly within the workspace."""
        if not rel_path or not isinstance(rel_path, str):
            return None, "Path must be a non-empty string."

        path_obj = Path(rel_path)
        if path_obj.is_absolute():
            return None, f"Absolute paths are rejected for containment: '{rel_path}'"

        try:
            target = (self.workspace / path_obj).resolve()
        except Exception as exc:
            return None, f"Failed to resolve path '{rel_path}': {exc}"

        try:
            target.relative_to(self.workspace)
        except ValueError:
            return None, f"Path traversal rejected: '{rel_path}' resolves outside the workspace."

        if must_exist and not target.exists():
            return None, f"File or directory does not exist: '{rel_path}'"

        return target, None

    def _clean_env(self) -> dict[str, str]:
        """Return a clean environment dict with sensitive credentials and keys removed."""
        env = os.environ.copy()
        sensitive_patterns = ["GATEWAY", "KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "CREDENTIAL"]
        for k in list(env.keys()):
            if any(pat in k.upper() for pat in sensitive_patterns):
                del env[k]

        env["PYTHONPATH"] = str(self.workspace / "src")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _bound_result(self, result: ToolResult) -> ToolResult:
        """Enforce MAX_OUTPUT_CHARS limit on result output."""
        if not result.ok:
            return result

        raw_str = result.output if isinstance(result.output, str) else json.dumps(result.output, ensure_ascii=False)
        if len(raw_str) > MAX_OUTPUT_CHARS:
            truncated_str = raw_str[:MAX_OUTPUT_CHARS] + "\n... [TRUNCATED: output exceeded max characters]"
            return ToolResult(
                ok=True,
                output=truncated_str,
                truncated=True,
            )
        return result

    # Tool implementations
    def _tool_list_files(self, path: str = ".", pattern: str | None = None) -> ToolResult:
        target_dir, err = self._resolve_safe_path(path, must_exist=True)
        if err:
            return ToolResult(ok=False, error=err)
        if not target_dir.is_dir():
            return ToolResult(ok=False, error=f"Not a directory: '{path}'")

        try:
            files: list[str] = []
            if pattern:
                entries = target_dir.glob(pattern)
            else:
                entries = target_dir.rglob("*")

            for p in entries:
                if p.is_file():
                    try:
                        rel = p.relative_to(self.workspace).as_posix()
                        files.append(rel)
                    except ValueError:
                        continue
                    if len(files) >= MAX_LIST_ENTRIES:
                        break

            files.sort()
            return ToolResult(ok=True, output=files)
        except Exception as exc:
            return ToolResult(ok=False, error=f"Error listing files: {exc}")

    def _tool_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
        target_file, err = self._resolve_safe_path(path, must_exist=True)
        if err:
            return ToolResult(ok=False, error=err)
        if not target_file.is_file():
            return ToolResult(ok=False, error=f"Not a regular file: '{path}'")

        try:
            content = target_file.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            total_lines = len(lines)
            s_line = start_line if start_line is not None else 1
            e_line = end_line if end_line is not None else total_lines

            if s_line < 1:
                return ToolResult(ok=False, error="start_line must be >= 1")
            if e_line < s_line:
                return ToolResult(ok=False, error=f"end_line ({e_line}) cannot be less than start_line ({s_line})")

            # Slice lines
            selected_lines = lines[s_line - 1 : e_line]
            truncated = False
            if len(selected_lines) > MAX_READ_LINES:
                selected_lines = selected_lines[:MAX_READ_LINES]
                truncated = True

            numbered = [f"{s_line + i:4d} | {line}" for i, line in enumerate(selected_lines)]
            output_text = "\n".join(numbered)
            return ToolResult(ok=True, output=output_text, truncated=truncated)
        except Exception as exc:
            return ToolResult(ok=False, error=f"Error reading file '{path}': {exc}")

    def _tool_search_code(self, pattern: str, glob: str | None = None, max_results: int | None = None) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as re_err:
            return ToolResult(ok=False, error=f"Invalid regex pattern '{pattern}': {re_err}")

        limit = max_results if max_results is not None else MAX_SEARCH_RESULTS
        limit = min(limit, MAX_SEARCH_RESULTS)

        matches: list[str] = []
        try:
            glob_pattern = glob if glob else "**/*"
            for p in self.workspace.glob(glob_pattern):
                if not p.is_file():
                    continue
                try:
                    rel = p.relative_to(self.workspace).as_posix()
                except ValueError:
                    continue

                # Don't search inside .git or cache dirs
                if "/." in f"/{rel}":
                    continue

                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    for line_idx, line in enumerate(text.splitlines(), start=1):
                        if regex.search(line):
                            matches.append(f"{rel}:{line_idx}: {line.strip()}")
                            if len(matches) >= limit:
                                break
                except Exception:
                    continue

                if len(matches) >= limit:
                    break

            return ToolResult(ok=True, output="\n".join(matches) if matches else "No matches found.")
        except Exception as exc:
            return ToolResult(ok=False, error=f"Search failed: {exc}")

    def _tool_run_tests(self, target: str | None = None) -> ToolResult:
        cmd = [sys.executable, "-m", "pytest"]

        if target:
            target_path_part = target.split("::")[0]
            _, err = self._resolve_safe_path(target_path_part, must_exist=True)
            if err:
                return ToolResult(ok=False, error=f"Invalid test target: {err}")
            cmd.append(target)

        report_file = self.workspace / ".pytest_report.xml"
        cmd.extend([f"--junitxml={report_file}", "-q", "--tb=short"])

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.workspace),
                env=self._clean_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_s,
            )

            # Parse junit xml report if generated
            summary: dict[str, Any] = {"passed": 0, "failed": 0, "skipped": 0, "failures": []}
            if report_file.exists():
                try:
                    tree = ET.parse(report_file)
                    root = tree.getroot()
                    # root is <testsuite> or <testsuites>
                    suite = root if root.tag == "testsuite" else root.find("testsuite")
                    if suite is not None:
                        summary["total"] = int(suite.attrib.get("tests", 0))
                        summary["failed"] = int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0))
                        summary["skipped"] = int(suite.attrib.get("skipped", 0))
                        summary["passed"] = max(0, summary["total"] - summary["failed"] - summary["skipped"])

                        for tc in suite.findall("testcase"):
                            tc_name = f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}"
                            for fail in tc.findall("failure"):
                                summary["failures"].append({"test": tc_name, "message": (fail.attrib.get("message", "") or fail.text or "").strip()[:500]})
                            for err_el in tc.findall("error"):
                                summary["failures"].append({"test": tc_name, "message": (err_el.attrib.get("message", "") or err_el.text or "").strip()[:500]})
                except Exception:
                    pass
                finally:
                    report_file.unlink(missing_ok=True)
            else:
                summary["stdout"] = proc.stdout[:1000]
                summary["stderr"] = proc.stderr[:1000]
                summary["exit_code"] = proc.returncode

            return ToolResult(ok=True, output=summary)

        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error=f"Test run timed out after {self.timeout_s} seconds.")
        except Exception as exc:
            return ToolResult(ok=False, error=f"Failed to execute tests: {exc}")

    def _tool_run_cli(self, args: list[str]) -> ToolResult:
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            return ToolResult(ok=False, error="'args' must be a list of strings.")

        cmd = [sys.executable, "-m", "tinyledger"] + args
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.workspace),
                env=self._clean_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_s,
            )
            return ToolResult(
                ok=True,
                output={
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error=f"CLI run timed out after {self.timeout_s} seconds.")
        except Exception as exc:
            return ToolResult(ok=False, error=f"CLI execution error: {exc}")

    def _tool_edit_file(self, path: str, old_str: str, new_str: str) -> ToolResult:
        target_file, err = self._resolve_safe_path(path, must_exist=True)
        if err:
            return ToolResult(ok=False, error=err)

        # Containment check: edits are strictly permitted only under src/
        src_dir = (self.workspace / "src").resolve()
        try:
            target_file.relative_to(src_dir)
        except ValueError:
            return ToolResult(ok=False, error=f"Containment violation: 'edit_file' can only modify files under 'src/'. Denied for '{path}'.")

        try:
            content = target_file.read_text(encoding="utf-8")
            count = content.count(old_str)
            if count == 0:
                return ToolResult(ok=False, error=f"String not found in '{path}'. Make sure 'old_str' matches the exact content.")
            if count > 1:
                return ToolResult(ok=False, error=f"Ambiguous replacement in '{path}': 'old_str' matched {count} times. Provide more unique context.")

            new_content = content.replace(old_str, new_str, 1)
            target_file.write_text(new_content, encoding="utf-8")

            return ToolResult(ok=True, output=f"Successfully edited '{path}'.")
        except Exception as exc:
            return ToolResult(ok=False, error=f"Failed to edit file '{path}': {exc}")
