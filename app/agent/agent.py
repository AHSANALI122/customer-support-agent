"""LangChain tool-calling agent core (F5).

Uses a manual ReAct loop (LLM → tool → LLM) instead of LangGraph's
create_react_agent to avoid Groq tool-call JSON compatibility issues.
"""

import asyncio
import json
import logging
import re
import time
import uuid

import httpx
from groq import APIError as GroqAPIError, BadRequestError, RateLimitError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from sqlmodel import Session

from app.agent.prompts import build_system_prompt
from app.agent.tools import CUSTOMER_TOOLS, current_session_id
from app.config import settings
from app.db import engine
from app.models import ChatMessage, MessageRole, ToolCallLog

logger = logging.getLogger("app.agent")

HISTORY_LIMIT = 20
FALLBACK_MESSAGE = (
    "I'm having trouble connecting right now, please try again in a moment."
)
_GENERIC_ERROR_SSE = (
    'event: error\ndata: {"message": "Something went wrong, please try again."}\n\n'
)

# Built once at import time so tool lookup is O(1).
_TOOL_MAP = {t.name: t for t in CUSTOMER_TOOLS}

# llama-3.x on Groq intermittently emits its *native* function-call markup —
# `<function=name{...json...}</function>` — instead of the OpenAI tool-call JSON
# schema. Groq can't parse that into a tool call, so it returns HTTP 400
# `tool_use_failed` with the raw markup in the error's `failed_generation` field.
# These patterns parse that markup back into a real tool call so the turn can
# continue instead of being thrown away. The `>` after the name is optional
# because the model emits both `<function=name>{...}` and `<function=name{...}`.
_FUNCTION_CALL_RE = re.compile(
    r"<function\s*=\s*([a-zA-Z0-9_]+)\s*>?\s*(\{.*?\})\s*</function>",
    re.DOTALL,
)
_FUNCTION_CALL_RE_OPEN = re.compile(
    r"<function\s*=\s*([a-zA-Z0-9_]+)\s*>?\s*(\{.*\})",
    re.DOTALL,
)


class _ToolCapture:
    """Records tool names called during one turn for SSE tool_call events."""

    def __init__(self):
        self.tool_names: list[str] = []


def _make_plain_llm() -> ChatGroq:
    """Fresh ChatGroq instance with NO tools bound.

    Used to force a final text answer when the model gets stuck repeating tool
    calls: without tools available it can only respond with prose, which makes
    it synthesise an answer from the tool results already in the conversation.
    """
    return ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=0.3,
        request_timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


def _make_llm():
    """Fresh ChatGroq instance with tools bound and parallel calls disabled."""
    return _make_plain_llm().bind_tools(CUSTOMER_TOOLS, parallel_tool_calls=False)


def _extract_text(content) -> str:
    """Flatten a message's content (str or list-of-parts) into plain text."""
    if isinstance(content, list):
        return "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content)


def _execute_tool(name: str, args: dict, session_id: uuid.UUID | None) -> str:
    """Run one tool by name, logging it for analytics. Never raises."""
    try:
        with Session(engine) as db:
            db.add(ToolCallLog(session_id=session_id, tool_name=name))
            db.commit()
    except Exception:
        logger.warning("Could not log tool call %s to DB", name)

    logger.info("[TOOL START] %s | args=%r", name, args)
    t0 = time.monotonic()

    fn = _TOOL_MAP.get(name)
    if fn is not None:
        try:
            result = str(fn.invoke(args))
        except Exception:
            logger.exception("Tool %s raised an exception", name)
            result = f"Tool {name} encountered an error; please try again."
    else:
        result = f"Unknown tool: {name}"

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info("[TOOL END] %s | %dms | %r", name, elapsed_ms, result[:200])
    return result


def _force_final_answer(msgs: list, session_id: uuid.UUID | None) -> str:
    """Make a final tool-less LLM call so the model must answer with prose.

    Called when the agent loops on repeated tool calls or exhausts its
    iteration budget but the tool results needed to answer are already present.
    """
    try:
        response = _call_with_retry(_make_plain_llm().invoke, msgs)
        text = _extract_text(response.content).strip()
        if text:
            return text
    except Exception:
        logger.exception("Forced final answer failed")
    # Genuinely couldn't produce an answer — fall back to a human (F7/F13).
    if session_id:
        return _escalate_on_loop(session_id)
    return FALLBACK_MESSAGE


def _call_with_retry(fn, *args, **kwargs):
    """Call fn once, retry once on transient Groq/network failures.

    A 400 BadRequestError is *not* transient — the request payload is the
    problem, not the connection — so it propagates immediately for the caller to
    handle (e.g. tool_use_failed recovery) rather than being retried blindly.
    """
    try:
        return fn(*args, **kwargs)
    except httpx.TimeoutException:
        logger.warning("LLM call timed out, retrying once…")
        time.sleep(1)
        return fn(*args, **kwargs)
    except httpx.ConnectError:
        logger.warning("LLM connection error, retrying once…")
        time.sleep(1)
        return fn(*args, **kwargs)
    except RateLimitError:
        logger.warning("LLM rate-limited (429), backing off and retrying…")
        time.sleep(2)
        return fn(*args, **kwargs)
    except BadRequestError:
        raise
    except GroqAPIError as exc:
        logger.warning("Groq API error (%s), retrying once…", exc)
        time.sleep(1)
        return fn(*args, **kwargs)


def _extract_failed_generation(exc: BadRequestError) -> str | None:
    """Pull `failed_generation` out of a Groq tool_use_failed error, or None.

    Returns None for any other 400 so unrelated bad requests still surface.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        try:
            body = exc.response.json()
        except Exception:
            return None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("code") == "tool_use_failed":
        failed = error.get("failed_generation")
        return failed if isinstance(failed, str) else None
    return None


def _recover_tool_calls(failed_generation: str) -> list[dict]:
    """Parse Llama's native `<function=name{...}</function>` markup into the
    tool-call dict shape LangChain expects. Returns [] if nothing parses."""
    matches = _FUNCTION_CALL_RE.findall(failed_generation)
    if not matches:
        matches = _FUNCTION_CALL_RE_OPEN.findall(failed_generation)

    tool_calls: list[dict] = []
    for name, raw_args in matches:
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            logger.warning("Could not parse recovered args for %s: %r", name, raw_args)
            continue
        if not isinstance(args, dict):
            continue
        tool_calls.append(
            {
                "name": name,
                "args": args,
                "id": f"recovered_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
        )
    return tool_calls


def _invoke_llm(llm, msgs):
    """Invoke the tool-bound LLM, recovering from Groq's `tool_use_failed` 400.

    When llama emits its native function-call markup instead of OpenAI JSON,
    Groq returns a 400 with the raw markup in `failed_generation`. We parse it
    back into a proper AIMessage with tool_calls so the loop continues — instead
    of failing the whole turn the way a blind retry of the same prompt would.
    """
    try:
        return _call_with_retry(llm.invoke, msgs)
    except BadRequestError as exc:
        failed = _extract_failed_generation(exc)
        if failed is None:
            raise
        recovered = _recover_tool_calls(failed)
        if not recovered:
            logger.error("tool_use_failed but no tool call could be recovered")
            raise
        logger.warning(
            "Recovered %d tool call(s) from Groq tool_use_failed: %s",
            len(recovered),
            [tc["name"] for tc in recovered],
        )
        return AIMessage(content="", tool_calls=recovered)


def _escalate_on_loop(session_id: uuid.UUID) -> str:
    from app.agent.tools import _open_ticket
    try:
        with Session(engine) as db:
            msg = _open_ticket(
                db,
                session_id,
                "Agent exceeded maximum steps — human review needed",
            )
    except Exception:
        logger.exception("Failed to open escalation ticket for session %s", session_id)
        msg = "Please contact our support team directly."
    return f"I wasn't able to fully resolve your question automatically. {msg}"


def _load_history(session_id: uuid.UUID) -> list:
    from sqlmodel import select
    with Session(engine) as session:
        rows = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(HISTORY_LIMIT)
        ).all()

    result = []
    for row in reversed(rows):
        if row.role == MessageRole.user:
            result.append(HumanMessage(content=row.content))
        elif row.role == MessageRole.assistant:
            result.append(AIMessage(content=row.content))
    return result


def _run_tool_loop(
    messages: list,
    capture: _ToolCapture | None = None,
) -> str:
    """Manual ReAct loop: call LLM, execute any tool calls, repeat.

    Replaces create_react_agent to eliminate LangGraph/Groq compatibility
    issues. Returns the final text reply, or FALLBACK_MESSAGE on failure.
    """
    llm = _make_llm()
    msgs = list(messages)
    session_id = current_session_id.get()

    # Cache each tool result by a (name, args) signature. Llama-on-Groq often
    # re-emits the identical tool call instead of answering; when it does we
    # serve the cached result instead of paying for the lookup again, and we
    # force a final answer so the turn can't spin until max_iterations.
    result_cache: dict[str, str] = {}

    for iteration in range(settings.agent_max_iterations):
        try:
            response = _invoke_llm(llm, msgs)
        except Exception:
            logger.exception("LLM invoke failed on iteration %d", iteration)
            return FALLBACK_MESSAGE

        msgs.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return _extract_text(response.content)

        looping = False
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            call_id = tc.get("id", "")

            if capture is not None:
                capture.tool_names.append(name)

            sig = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            if sig in result_cache:
                # Identical call already answered this turn — the model is
                # looping. Reuse the result and break out to a final answer.
                looping = True
                result = result_cache[sig]
                logger.info("[TOOL REPEAT] %s | reusing cached result", name)
            else:
                result = _execute_tool(name, args, session_id)
                result_cache[sig] = result

            msgs.append(ToolMessage(content=result, tool_call_id=call_id))

        if looping:
            return _force_final_answer(msgs, session_id)

    # Exhausted the iteration budget on distinct tool calls — make the model
    # answer from everything gathered rather than silently escalating.
    return _force_final_answer(msgs, session_id)


def run_agent(
    session_id: uuid.UUID | str,
    message: str,
    customer_email: str | None = None,
) -> str:
    """Run one agent turn and return the reply text. Never raises (F13)."""
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    current_session_id.set(session_id)

    messages = [SystemMessage(content=build_system_prompt(customer_email))]
    messages.extend(_load_history(session_id))
    messages.append(HumanMessage(content=message))

    try:
        return _run_tool_loop(messages)
    except Exception:
        logger.exception("run_agent unexpected failure for session %s", session_id)
        return FALLBACK_MESSAGE


async def stream_agent(
    session_id: uuid.UUID | str,
    message: str,
    customer_email: str | None = None,
):
    """Async generator yielding SSE strings for one agent turn.

    Runs the tool loop in a thread (reliable), then fake-streams the reply
    word-by-word so the frontend shows a typing effect.

    Events: token | tool_call | done | error
    """
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    current_session_id.set(session_id)

    messages = [SystemMessage(content=build_system_prompt(customer_email))]
    messages.extend(_load_history(session_id))
    messages.append(HumanMessage(content=message))

    capture = _ToolCapture()

    try:
        reply = await asyncio.to_thread(_run_tool_loop, messages, capture)
    except Exception:
        logger.exception("stream_agent failed for session %s", session_id)
        yield _GENERIC_ERROR_SSE
        return

    if reply == FALLBACK_MESSAGE:
        yield _GENERIC_ERROR_SSE
        return

    # Emit tool_call events first so the frontend shows a "checking…" indicator
    # before any text appears.
    for tool_name in capture.tool_names:
        yield f'event: tool_call\ndata: {json.dumps({"tool": tool_name})}\n\n'

    # Fake-stream the reply word-by-word for a natural typing effect.
    words = reply.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f'event: token\ndata: {json.dumps({"text": chunk})}\n\n'
        await asyncio.sleep(0.02)

    yield (
        f'event: done\ndata: {json.dumps({"session_id": str(session_id), "full_reply": reply})}\n\n'
    )
