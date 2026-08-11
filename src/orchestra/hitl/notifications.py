"""Pluggable notification seam for the approval queue — same pattern as `tools/implementations/
web_search.py`'s `set_search_provider`: a default that's honest about not being a real
integration, and a seam a real one (email, Slack, whatever) can be dropped into later without
touching the escalation/queue code that calls it.
"""

import logging
from typing import Callable

from orchestra.hitl.approval_queue import ApprovalRequest

logger = logging.getLogger("orchestra.hitl")

_notifier: Callable[[ApprovalRequest], None] | None = None


def set_notifier(notifier: Callable[[ApprovalRequest], None]) -> None:
    global _notifier
    _notifier = notifier


def _default_notifier(request: ApprovalRequest) -> None:
    logger.warning(
        "Approval needed [%s]: task_id=%s reason=%s step=%s — review with "
        "`python scripts/review_queue.py show %s`",
        request.id,
        request.task_id,
        request.reason,
        request.current_step_id,
        request.id,
    )


def notify_reviewer(request: ApprovalRequest) -> None:
    (_notifier or _default_notifier)(request)
