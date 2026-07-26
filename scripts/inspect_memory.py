"""Inspect a task's Redis-backed working memory from outside the process that ran it.

This is the point of Phase 2.1: the memory store isn't a log printed to one terminal, it's a
shared store any process can attach to while a task is in flight. Requires `docker-compose up
-d` (Redis) and a task_id — print one by running scripts/run_task.py, which logs it, or check
an escalated task (its memory is deliberately NOT cleared, since a human still needs it).

Usage:
    python scripts/inspect_memory.py <task_id>
"""

import sys

from dotenv import load_dotenv

from orchestra.memory.working_memory import WorkingMemory

load_dotenv()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_memory.py <task_id>")
        raise SystemExit(1)

    memory = WorkingMemory(sys.argv[1])

    print("=== TASK ===")
    print(memory.get_task())

    plan = memory.get_plan()
    print("\n=== PLAN ===")
    if plan is None:
        print("(no plan recorded yet)")
    else:
        for step in plan.steps:
            print(f"  [{step.domain}] {step.step_id}: {step.description}")

    print("\n=== COMPLETED OUTPUTS ===")
    completed = memory.get_completed_outputs()
    if not completed:
        print("(none yet)")
    for step_id, result in completed.items():
        print(f"  {step_id} (attempt {result.attempt}): {result.output[:200]}")

    if plan is not None:
        print("\n=== INTERMEDIATE RESULTS (all attempts, per step) ===")
        for step in plan.steps:
            attempts = memory.get_intermediate_results(step.step_id)
            if attempts:
                print(f"  {step.step_id}: {len(attempts)} attempt(s)")

    print("\n=== ERROR LOG ===")
    errors = memory.get_error_log()
    if not errors:
        print("(none)")
    for entry in errors:
        print(f"  [{entry['timestamp']}] {entry.get('step_id')}: {entry['message']}")


if __name__ == "__main__":
    main()
