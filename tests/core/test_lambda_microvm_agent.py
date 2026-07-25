import os
import runpy
import tempfile
import time
from pathlib import Path
from typing import Any


def load_agent(working_dir: str, process_dir: str) -> dict[str, Any]:
    previous_working_dir = os.environ.get("LM_DELUGE_WORKING_DIR")
    previous_process_dir = os.environ.get("LM_DELUGE_PROCESS_DIR")
    os.environ["LM_DELUGE_WORKING_DIR"] = working_dir
    os.environ["LM_DELUGE_PROCESS_DIR"] = process_dir
    try:
        return runpy.run_path(
            "src/lm_deluge/tool/prefab/sandbox/lambda_microvm_agent.py"
        )
    finally:
        if previous_working_dir is None:
            os.environ.pop("LM_DELUGE_WORKING_DIR", None)
        else:
            os.environ["LM_DELUGE_WORKING_DIR"] = previous_working_dir
        if previous_process_dir is None:
            os.environ.pop("LM_DELUGE_PROCESS_DIR", None)
        else:
            os.environ["LM_DELUGE_PROCESS_DIR"] = previous_process_dir


def test_foreground_execution(agent: dict[str, Any]):
    result = agent["_run_foreground"]("printf hello", 5000)
    assert result == {
        "stdout": "hello",
        "stderr": "",
        "exitCode": 0,
        "timedOut": False,
    }


def test_timeout(agent: dict[str, Any]):
    result = agent["_run_foreground"]("sleep 1", 10)
    assert result["timedOut"] is True
    assert result["exitCode"] != 0


def test_background_execution(agent: dict[str, Any]):
    result = agent["_run_background"]("printf background", "worker")
    assert result["name"] == "worker"
    assert isinstance(result["pid"], int)

    deadline = time.monotonic() + 2
    details = agent["_list_processes"]("worker")
    while details[0]["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
        details = agent["_list_processes"]("worker")

    assert details[0]["running"] is False
    assert details[0]["exitCode"] == 0
    assert details[0]["recentOutput"] == "background"


def main():
    source = Path(
        "src/lm_deluge/tool/prefab/sandbox/lambda_microvm_agent.py"
    ).read_text()
    assert source.startswith("from __future__ import annotations\n")
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        working_dir = root / "workspace"
        process_dir = root / "processes"
        working_dir.mkdir()
        process_dir.mkdir()
        agent = load_agent(str(working_dir), str(process_dir))
        test_foreground_execution(agent)
        test_timeout(agent)
        test_background_execution(agent)
    print("Lambda MicroVM agent tests passed")


if __name__ == "__main__":
    main()
