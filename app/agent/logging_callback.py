import logging
import time
from typing import Any, Dict
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from sqlmodel import Session

from app.db import engine
from app.models import ToolCallLog

logger = logging.getLogger("app.agent")


class ToolLoggingCallback(BaseCallbackHandler):
    """Logs every tool call with name, input, output, and elapsed time."""

    def __init__(self):
        self._start_times: dict[UUID, float] = {}

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.monotonic()
        tool_name = serialized.get("name", "unknown")
        logger.info("[TOOL START] %s | input=%r", tool_name, input_str)
        # Inline import avoids a circular dependency (tools → logging_callback → tools).
        from app.agent.tools import current_session_id
        with Session(engine) as db:
            db.add(ToolCallLog(session_id=current_session_id.get(), tool_name=tool_name))
            db.commit()

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._start_times.pop(run_id, time.monotonic())
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "[TOOL END] run_id=%s | elapsed=%dms | output=%r",
            run_id,
            elapsed_ms,
            str(output)[:200],
        )

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._start_times.pop(run_id, None)
        logger.error("[TOOL ERROR] run_id=%s | error=%s", run_id, error)
