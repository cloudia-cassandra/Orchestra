"""Sandboxed(-ish) Python code execution.

Runs in a throwaway temp directory with a wall-clock timeout and no access to the calling
process's working directory. It is process-isolated, NOT network- or resource-isolated —
hardening with a container (Docker/gVisor/firejail) is a follow-up before this ever runs
untrusted, internet-facing input.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def code_execution(code: str, timeout_seconds: int = 10) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            stderr = (exc.stderr or "") + f"\n[timed out after {timeout_seconds}s]"
            return {"stdout": exc.stdout or "", "stderr": stderr, "exit_code": -1}
