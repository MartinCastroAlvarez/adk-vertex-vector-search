"""Run the agent: ADK ``Runner`` + ``InMemorySessionService``, one question per call.

:meth:`Researcher.ask` returns the reply plus what happened on the way - the searches
(query, filters, hit ids), the hop count and the citations the guard accepted - read
from the per-invocation trace in session state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agentic_rag import trace as tr
from agentic_rag.config import APP_NAME
from agentic_rag.config import Settings
from agentic_rag.researcher import build_agent
from agentic_rag.researcher import retriever_handle


@dataclass
class Answer:
    question: str
    reply: str
    status: str
    citations: list[str] = field(default_factory=list)
    searches: list[dict[str, Any]] = field(default_factory=list)
    hops: int = 0
    events: list[Event] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (part.function_call.name or "", dict(part.function_call.args or {}))
            for event in self.events
            for part in (event.content.parts or [] if event.content else [])
            if part.function_call
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "reply": self.reply,
            "status": self.status,
            "citations": self.citations,
            "hops": self.hops,
            "searches": self.searches,
        }


class Researcher:
    """A question-answering front-end over one agent and one session service."""

    def __init__(self, agent: LlmAgent | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.agent = agent or build_agent(self.settings)
        self.sessions = InMemorySessionService()
        self.runner = Runner(agent=self.agent, app_name=APP_NAME, session_service=self.sessions)
        # A backstop on top of the hop budget: searches + answer + a little slack.
        self.run_config = RunConfig(max_llm_calls=2 * self.settings.max_hops + 4)

    def close(self) -> None:
        handle = retriever_handle(self.agent)
        if handle is not None:
            handle.close()

    async def ask(
        self, question: str, user_id: str = "local", session_id: str | None = None
    ) -> Answer:
        session_id = session_id or uuid.uuid4().hex
        existing = await self.sessions.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        if existing is None:
            await self.sessions.create_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )
        content = types.Content(role="user", parts=[types.Part(text=question)])
        events = [
            event
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
                run_config=self.run_config,
            )
        ]
        reply = ""
        for event in events:
            if event.author != "user" and event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts if not p.thought)
                reply = text or reply
        session = await self.sessions.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        state = dict(session.state.get(tr.TRACE_KEY) or {}) if session else {}
        answer = state.get("answer") or {}
        return Answer(
            question=question,
            reply=reply.strip(),
            status=str(answer.get("status", "unknown")),
            citations=list(answer.get("citations", [])),
            searches=list(state.get("searches", [])),
            hops=int(state.get("hops", 0)),
            events=events,
        )
