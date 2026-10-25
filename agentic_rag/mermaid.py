"""A Mermaid flowchart generated from the real agent and the configured store."""

from __future__ import annotations

from google.adk.agents import LlmAgent

from agentic_rag.config import Settings
from agentic_rag.researcher import build_agent
from agentic_rag.stub import ScriptedLlm

_STORES = {
    "local": "LocalVectorSearch<br/>FAISS dense · BM25 sparse · RRF · restricts · crowding",
    "vertex": "VertexVectorSearchStore<br/>Vertex AI Vector Search · find_neighbors",
}


def _name(callback: object) -> str:
    callbacks = callback if isinstance(callback, list) else [callback]
    return ", ".join(getattr(c, "__name__", type(c).__name__) for c in callbacks if c)


def to_mermaid(settings: Settings, agent: LlmAgent | None = None) -> str:
    agent = agent or build_agent(settings, model=ScriptedLlm(model="graph"))
    tools = [str(getattr(t, "name", None) or getattr(t, "__name__", "tool")) for t in agent.tools]
    return "\n".join(
        [
            "flowchart LR",
            '  q(["question"])',
            f'  subgraph agent ["LlmAgent {agent.name}"]',
            "    direction TB",
            f'    hint["<b>before_model: {_name(agent.before_model_callback)}</b><br/>'
            'catalog + filters read off the question"]',
            '    llm["model<br/>plan · rewrite · self-check"]',
            f'    tool["<b>{", ".join(tools)}</b><br/>validate filters · hop budget '
            f'{settings.max_hops}<br/>neutralise passages"]',
            f'    guard["<b>after_model: {_name(agent.after_model_callback)}</b><br/>'
            'citations ⊆ retrieved · else refuse"]',
            "    hint --> llm",
            '    llm -- "not enough yet: search again" --> tool',
            "    tool -- passages --> llm",
            "    llm --> guard",
            "  end",
            '  ret["Retriever<br/>dense + BM25 sparse query<br/>'
            'restricts · numeric filters · crowding"]',
            f'  store[("VectorStore<br/>{_STORES[settings.store]}")]',
            "  q --> hint",
            "  tool --> ret --> store",
            '  a(["answer with [citations] or refusal"])',
            "  guard --> a",
        ]
    )
