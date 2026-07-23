"""Manual smoke test: runs a real task through the compiled graph.

Requires ANTHROPIC_API_KEY to be set (see .env.example). Usage:
    python scripts/run_task.py "Summarize the plot of Hamlet in three sentences."
"""

import sys

from dotenv import load_dotenv

from orchestra.orchestration.graph import build_graph

load_dotenv()


def main() -> None:
    task = " ".join(sys.argv[1:]) or "Summarize the plot of Hamlet in three sentences."
    graph = build_graph()
    final_state = graph.invoke({"task": task}, config={"recursion_limit": 50})

    print("\n=== PLAN ===")
    for step in final_state["plan"].steps:
        print(f"  [{step.domain}] {step.step_id}: {step.instruction}")

    print("\n=== STATUS ===")
    print(final_state["status"])

    print("\n=== FINAL OUTPUT ===")
    print(final_state.get("final_output"))


if __name__ == "__main__":
    main()
