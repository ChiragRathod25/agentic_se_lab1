"""Task 4: Native function calling agent loop for debugging tinyledger.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root and task directory to sys.path
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common.llm_client import LLMClient, message_to_dict
from mock_model import MockOpenAIClient
from reset_workspace import main as reset_workspace
from tools import ToolRegistry

WORKSPACE_DIR = HERE / "workspace"
LOGS_DIR = HERE / "logs"
TRACE_LOG = LOGS_DIR / "trace.jsonl"

TASK_PROMPT = "Make all tests pass and fix the problem described in ISSUE.md. Do not modify tests."
SYSTEM_PROMPT = (
    "You are an autonomous software engineering agent tasked with repairing a Python repository.\n"
    "Goal: Make all tests pass and fix the problem described in ISSUE.md. Do not modify tests.\n"
    "Guidelines:\n"
    "1. Explore the workspace with list_files, read_file, search_code, and run_tests.\n"
    "2. Check ISSUE.md to understand the reported bug.\n"
    "3. Run tests using run_tests to identify existing failures.\n"
    "4. Fix bugs strictly within src/ using edit_file. Tests must remain untouched.\n"
    "5. Verify all tests pass and that the issue is completely resolved.\n"
    "6. When finished, provide a concise summary of the bugs fixed."
)


def log_tool_trace(tool_name: str, arguments: str, result, duration_s: float) -> None:
    """Log one line per tool call to logs/trace.jsonl."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "arguments": arguments,
        "ok": result.ok,
        "error": result.error,
        "truncated": result.truncated,
        "duration_s": round(duration_s, 4),
    }
    with TRACE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_agent(is_mock: bool = False, max_steps: int = 35) -> None:
    if not WORKSPACE_DIR.exists():
        print("[agent] Workspace does not exist. Initializing fresh workspace...")
        reset_workspace()

    registry = ToolRegistry(workspace=WORKSPACE_DIR, timeout_s=60)
    tools = registry.schemas()

    # Clear previous trace log for fresh run
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if TRACE_LOG.exists():
        TRACE_LOG.unlink()

    if is_mock:
        print("[agent] Running with scripted MOCK model (offline security verification)...")
        llm = LLMClient(
            model="mock",
            log_path=str(LOGS_DIR / "mock_run.jsonl"),
            openai_client=MockOpenAIClient(),
        )
    else:
        print("[agent] Running with live UPB AI Gateway model...")
        llm = LLMClient(log_path=str(LOGS_DIR / "agent_run.jsonl"))

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TASK_PROMPT},
    ]

    print(f"[agent] Initial prompt: {TASK_PROMPT}\n")

    for step in range(1, max_steps + 1):
        print(f"--- Step {step} ---")
        message = llm.chat_with_tools(messages, tools=tools)
        messages.append(message_to_dict(message))

        tool_calls = getattr(message, "tool_calls", None)

        # If assistant has a message text, print it
        if message.content:
            print(f"[Assistant]: {message.content.strip()}\n")

        # If no tool calls, the model decided to finish
        if not tool_calls:
            print("[agent] Model finished conversation (no further tool calls).")
            break

        for call in tool_calls:
            name = call.function.name
            args_str = call.function.arguments
            print(f"[Tool Call] {name}({args_str})")

            start_t = time.perf_counter()
            result = registry.execute(name, args_str)
            duration = time.perf_counter() - start_t

            log_tool_trace(name, args_str, result, duration)

            if result.ok:
                summary = str(result.output)[:160].replace("\n", " ")
                print(f"  -> OK ({duration:.3f}s): {summary}...")
            else:
                print(f"  -> ERROR ({duration:.3f}s): {result.error}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result.to_content(),
                }
            )
        print()

    print("\n[agent] Run complete.")
    print(f"[agent] Trace saved to: {TRACE_LOG}")
    print(f"[agent] Total usage: {llm.usage_summary()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4 agent loop with function calling.")
    parser.add_argument("--mock", action="store_true", help="Run with deterministic mock model.")
    parser.add_argument("--steps", type=int, default=35, help="Maximum agent steps.")
    args = parser.parse_args()

    run_agent(is_mock=args.mock, max_steps=args.steps)


if __name__ == "__main__":
    main()
