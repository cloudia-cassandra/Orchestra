"""Memory dashboard (Phase 2.4): what Orchestra's long-term memory currently remembers about
one user — every stored record, its importance score, and a way to act on a data deletion
request. This is a maintenance/inspection view, not the Phase 3 human-in-the-loop review UI;
it lives under `ui/` for the same reason that one will — a Streamlit front end over otherwise
headless state.

Requires `docker-compose up -d` (ChromaDB).

Usage:
    streamlit run ui/memory_dashboard.py
"""

import streamlit as st
from dotenv import load_dotenv

from orchestra.memory.consolidation import MemoryConsolidator
from orchestra.memory.long_term_memory import LongTermMemory

load_dotenv()

st.set_page_config(page_title="Orchestra — Memory Dashboard", layout="wide")
st.title("Memory dashboard")
st.caption("What the system remembers about each user, and a way to act on a deletion request.")

user_id = st.text_input("User ID", value="default_user")

memory = LongTermMemory()

if not user_id:
    st.stop()

records = memory.list_for_user(user_id)

st.subheader(f"{len(records)} memory record(s) for `{user_id}`")

if not records:
    st.info("Nothing remembered about this user yet.")
else:
    for record in records:
        importance = memory.compute_importance(record)
        age = memory.age_days(record)
        badge = "🧩 consolidated" if record.consolidated else record.outcome
        with st.expander(
            f"[{badge}] {record.task}  ·  importance {importance:.2f}  ·  "
            f"accessed {record.access_count}x  ·  last touched {age:.1f}d ago"
        ):
            st.markdown(f"**Approach:** {record.approach_summary}")
            if record.plan_steps:
                st.markdown("**Plan used:**")
                st.write(record.plan_steps)
            if record.domain_facts:
                st.markdown("**Facts discovered:**")
                st.write(record.domain_facts)
            if record.user_preferences:
                st.markdown("**Preferences observed:**")
                st.write(record.user_preferences)
            if record.consolidated:
                st.caption(f"Merged from {record.source_count} original record(s).")
            st.caption(f"task_id: {record.task_id}  ·  stored: {record.completed_at}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Consolidate")
    st.caption("Merge near-duplicate memories for this user into higher-level summaries.")
    if st.button("Run consolidation", disabled=not records):
        merged = MemoryConsolidator(memory=memory).consolidate(user_id)
        if merged:
            st.success(f"Merged into {len(merged)} consolidated record(s). Refresh to see them.")
        else:
            st.info("Nothing similar enough to merge.")

with col2:
    st.subheader("Delete user data")
    st.caption("Permanently erase every memory record for this user (data deletion request).")
    confirm = st.checkbox(f"I confirm deletion of all memory for `{user_id}`")
    if st.button("Delete all memory for this user", disabled=not confirm, type="primary"):
        deleted = memory.delete_for_user(user_id)
        st.success(f"Deleted {deleted} record(s) for `{user_id}`.")
        st.rerun()
