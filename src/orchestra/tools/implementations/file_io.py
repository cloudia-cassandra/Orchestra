"""Sandboxed file read/write, confined to a single workspace directory."""

import os
from pathlib import Path


def _workspace_root() -> Path:
    root = Path(os.environ.get("ORCHESTRA_WORKSPACE_DIR", "workspace")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def file_read_write(operation: str, path: str, content: str | None = None) -> dict:
    root = _workspace_root()
    target = (root / path).resolve()

    if target != root and root not in target.parents:
        raise ValueError(f"Path {path!r} escapes the sandboxed workspace.")

    if operation == "read":
        if not target.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        return {"content": target.read_text(encoding="utf-8"), "bytes_written": None}

    if operation == "write":
        if content is None:
            raise ValueError("content is required for a write operation.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"content": None, "bytes_written": len(content.encode("utf-8"))}

    raise ValueError(f"Unknown operation: {operation!r} (expected 'read' or 'write')")
