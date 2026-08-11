"""Phase 3.2: review and decide on pending human-in-the-loop escalations from the terminal.

Requires `docker-compose up -d` (Postgres — the approval queue's backing store; Redis, for
working memory the resumed task will touch).

Usage:
    python scripts/review_queue.py list [user_id]
    python scripts/review_queue.py show <request_id>
    python scripts/review_queue.py approve <request_id> [--notes "..."]
    python scripts/review_queue.py reject <request_id> --notes "why"
    python scripts/review_queue.py modify <request_id> --output "corrected output" [--notes "..."]
"""

import argparse

from dotenv import load_dotenv

from orchestra.hitl.approval_queue import ApprovalQueue
from orchestra.hitl.resume import resume_task

load_dotenv()

REVIEWED_BY = "cli"  # no auth system yet — see intake_node's DEFAULT_USER_ID for the same honesty


def cmd_list(args: argparse.Namespace) -> None:
    requests = ApprovalQueue().list_pending(user_id=args.user_id)
    if not requests:
        print("Nothing pending.")
        return
    for r in requests:
        print(f"{r.id}  [{r.reason}]  task={r.task_id}  user={r.user_id}  step={r.current_step_id}")


def cmd_show(args: argparse.Namespace) -> None:
    request = ApprovalQueue().get(args.request_id)
    if request is None:
        print(f"No request found for id={args.request_id!r}")
        raise SystemExit(1)

    print(f"=== {request.id} ({request.status}) ===")
    print(f"Reason: {request.reason}")
    print(f"Task: {request.original_task}")
    print(f"Step needing a decision: {request.current_step_id or '(plan-level — nothing has run yet)'}")
    print("\n--- Plan ---")
    for step in request.state_snapshot["plan"]["steps"]:
        print(f"  [{step['domain']}] {step['step_id']}: {step['description']}")
    print("\n--- Completed so far ---")
    completed = request.state_snapshot["specialist_results"]
    if not completed:
        print("  (none)")
    for r in completed:
        print(f"  {r['step_id']}: {r['output'][:200]}")
    print("\n--- Proposed action awaiting a decision ---")
    print(f"  {request.proposed_action or '(none — this is a plan-level escalation)'}")
    print(f"\nEscalation detail: {request.escalation_detail}")


def cmd_decide(args: argparse.Namespace) -> None:
    queue = ApprovalQueue()
    queue.decide(
        args.request_id,
        decision=args.decision,
        decided_by=REVIEWED_BY,
        reviewer_notes=args.notes,
        modified_output=getattr(args, "output", None),
    )
    result = resume_task(args.request_id, queue=queue)
    print(f"Applied '{args.decision}'. Task now: status={result.get('status')!r}")
    if result.get("status") == "complete":
        print(f"Final output:\n{result.get('final_output')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("user_id", nargs="?", default=None)
    list_parser.set_defaults(func=cmd_list)

    show_parser = sub.add_parser("show")
    show_parser.add_argument("request_id")
    show_parser.set_defaults(func=cmd_show)

    for name, decision in (("approve", "approved"), ("reject", "rejected"), ("modify", "modified")):
        p = sub.add_parser(name)
        p.add_argument("request_id")
        p.add_argument("--notes", default=None)
        if decision == "modified":
            p.add_argument("--output", required=True, dest="output")
        p.set_defaults(func=cmd_decide, decision=decision)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
