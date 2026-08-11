"""Approval queue dashboard (Phase 3.2): review paused, escalated tasks and decide — approve,
reject, or modify — without leaving the browser. Deciding here immediately resumes the task
(hitl.resume.resume_task), same as `scripts/review_queue.py`'s decide commands.

Requires `docker-compose up -d` (Postgres, Redis).

Usage:
    streamlit run ui/approval_dashboard.py
"""

import streamlit as st
from dotenv import load_dotenv

from orchestra.hitl.approval_queue import ApprovalQueue
from orchestra.hitl.resume import resume_task

load_dotenv()

REVIEWED_BY = "dashboard"  # no auth system yet — same honesty as intake_node's DEFAULT_USER_ID

st.set_page_config(page_title="Orchestra — Approval Queue", layout="wide")
st.title("Approval queue")
st.caption("Tasks paused on a Phase 3.1 escalation trigger, waiting on a human decision.")

queue = ApprovalQueue()
user_filter = st.text_input("Filter by user_id (blank = everyone)", value="")
pending = queue.list_pending(user_id=user_filter or None)

st.subheader(f"{len(pending)} pending")

if not pending:
    st.info("Nothing waiting on a decision.")

for request in pending:
    with st.expander(f"[{request.reason}] {request.original_task}  ·  {request.id}"):
        st.markdown(f"**User:** `{request.user_id}`  ·  **Task ID:** `{request.task_id}`")
        st.markdown(
            f"**Step needing a decision:** "
            f"{request.current_step_id or '_(plan-level — nothing has run yet)_'}"
        )

        st.markdown("**Plan:**")
        for step in request.state_snapshot["plan"]["steps"]:
            st.write(f"[{step['domain']}] {step['step_id']}: {step['description']}")

        completed = request.state_snapshot["specialist_results"]
        if completed:
            st.markdown("**Completed so far:**")
            for r in completed:
                st.write(f"{r['step_id']}: {r['output']}")

        if request.proposed_action:
            st.markdown("**Proposed action awaiting a decision:**")
            st.code(request.proposed_action)

        st.markdown(f"**Escalation detail:** `{request.escalation_detail}`")

        is_plan_level = request.current_step_id is None
        notes = st.text_area("Reviewer notes", key=f"notes-{request.id}")

        cols = st.columns(3)
        if cols[0].button("Approve", key=f"approve-{request.id}"):
            queue.decide(request.id, "approved", REVIEWED_BY, reviewer_notes=notes)
            result = resume_task(request.id, queue=queue)
            st.success(f"Approved. Task status: {result.get('status')}")
            st.rerun()

        if cols[1].button("Reject", key=f"reject-{request.id}", disabled=not notes):
            queue.decide(request.id, "rejected", REVIEWED_BY, reviewer_notes=notes)
            result = resume_task(request.id, queue=queue)
            st.warning(f"Rejected. Task status: {result.get('status')}")
            st.rerun()
        if is_plan_level:
            cols[1].caption("Notes required — this ends the task, no retry.")

        if not is_plan_level:
            modified_output = st.text_area(
                "Modified output (only used by Modify)", key=f"modified-{request.id}"
            )
            if cols[2].button("Modify", key=f"modify-{request.id}", disabled=not modified_output):
                queue.decide(
                    request.id,
                    "modified",
                    REVIEWED_BY,
                    reviewer_notes=notes,
                    modified_output=modified_output,
                )
                result = resume_task(request.id, queue=queue)
                st.success(f"Modified output accepted. Task status: {result.get('status')}")
                st.rerun()
        else:
            cols[2].caption("Modify isn't supported for a whole plan yet.")
