"""Registers the Phase 1.3 tool set into a module-level ToolRegistry singleton."""

from orchestra.tools.implementations.api_call import api_call
from orchestra.tools.implementations.code_execution import code_execution
from orchestra.tools.implementations.database import database_query
from orchestra.tools.implementations.file_io import file_read_write
from orchestra.tools.implementations.web_search import web_search
from orchestra.tools.registry import RateLimit, ToolRegistry, ToolSpec

registry = ToolRegistry()

registry.register(
    ToolSpec(
        name="web_search",
        description="Search the web and return ranked results with titles, URLs, and snippets.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                        },
                    },
                }
            },
        },
        allowed_domains=["research"],
        rate_limit=RateLimit(max_calls=10, per_seconds=60),
        handler=web_search,
    )
)

registry.register(
    ToolSpec(
        name="file_read_write",
        description="Read or write a text file inside the sandboxed workspace directory.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["read", "write"]},
                "path": {"type": "string"},
                "content": {"type": ["string", "null"]},
            },
            "required": ["operation", "path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "content": {"type": ["string", "null"]},
                "bytes_written": {"type": ["integer", "null"]},
            },
        },
        allowed_domains=["research", "data_analysis", "writing", "code_execution"],
        rate_limit=RateLimit(max_calls=30, per_seconds=60),
        handler=file_read_write,
    )
)

registry.register(
    ToolSpec(
        name="code_execution",
        description="Run a Python snippet in an isolated process and return stdout/stderr/exit code.",
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout_seconds": {"type": "integer", "default": 10},
            },
            "required": ["code"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "exit_code": {"type": "integer"},
            },
        },
        allowed_domains=["code_execution", "data_analysis"],
        rate_limit=RateLimit(max_calls=10, per_seconds=60),
        handler=code_execution,
    )
)

registry.register(
    ToolSpec(
        name="database_query",
        description="Run a read-only SELECT query against the Orchestra Postgres database.",
        input_schema={
            "type": "object",
            "properties": {
                "sql": {"type": "string"},
                "params": {"type": "array", "items": {}},
            },
            "required": ["sql"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "rows": {"type": "array", "items": {"type": "object"}},
                "row_count": {"type": "integer"},
            },
        },
        allowed_domains=["data_analysis"],
        rate_limit=RateLimit(max_calls=20, per_seconds=60),
        handler=database_query,
    )
)

registry.register(
    ToolSpec(
        name="api_call",
        description="Make an outbound HTTP GET or POST request to an allowlisted host.",
        input_schema={
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["GET", "POST"]},
                "url": {"type": "string"},
                "headers": {"type": "object"},
                "json_body": {"type": "object"},
                "timeout_seconds": {"type": "integer", "default": 10},
            },
            "required": ["method", "url"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "status_code": {"type": "integer"},
                "body": {"type": "string"},
            },
        },
        allowed_domains=["research", "data_analysis", "code_execution"],
        rate_limit=RateLimit(max_calls=20, per_seconds=60),
        handler=api_call,
    )
)
