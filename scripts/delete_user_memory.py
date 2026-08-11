"""Delete everything Orchestra's long-term memory remembers about one user.

This is the "delete endpoint" Phase 2.4 asks for, honestly scoped to where the project actually
is: there's no HTTP server yet (that's a later ticket), so this is the callable surface a data
request gets routed to today — `LongTermMemory.delete_for_user`, wrapped in a script the way
Phase 2.1's `scripts/inspect_memory.py` wraps working-memory reads. The memory dashboard
(`ui/memory_dashboard.py`) calls the same underlying method from its delete button.

Requires `docker-compose up -d` (ChromaDB).

Usage:
    python scripts/delete_user_memory.py <user_id>
"""

import sys

from dotenv import load_dotenv

from orchestra.memory.long_term_memory import LongTermMemory

load_dotenv()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/delete_user_memory.py <user_id>")
        raise SystemExit(1)

    user_id = sys.argv[1]
    deleted = LongTermMemory().delete_for_user(user_id)
    print(f"Deleted {deleted} memory record(s) for user_id={user_id!r}.")


if __name__ == "__main__":
    main()
