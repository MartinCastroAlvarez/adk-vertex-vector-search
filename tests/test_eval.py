"""ADK's own evaluator (``AgentEvaluator``, what ``adk eval`` runs) on the stub agent."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_set import EvalSet

EVALS = Path(__file__).resolve().parent.parent / "evals"
EVALSET = EVALS / "agentic_rag.evalset.json"


@pytest.fixture
def stub_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_PROVIDER", "stub")
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path))
    importlib.reload(importlib.import_module("agentic_rag.agent"))


def _eval_set() -> EvalSet:
    return EvalSet.model_validate_json(EVALSET.read_text())


async def test_eval_set_passes(stub_agent: None) -> None:
    config = AgentEvaluator.find_config_for_test_file(str(EVALSET))
    assert set(config.criteria) == {"tool_trajectory_avg_score", "response_match_score"}
    await AgentEvaluator.evaluate_eval_set(
        agent_module="agentic_rag.agent",
        eval_set=_eval_set(),
        eval_config=config,
        num_runs=1,
        print_detailed_results=False,
    )


async def test_eval_catches_a_wrong_filter(stub_agent: None) -> None:
    eval_set = _eval_set()
    uses = eval_set.eval_cases[0].conversation[0].intermediate_data.tool_uses  # type: ignore[union-attr]
    uses[0].args = {**uses[0].args, "version": "3.0"}  # type: ignore[index]
    config = AgentEvaluator.find_config_for_test_file(str(EVALSET))
    with pytest.raises(AssertionError, match="tool_trajectory_avg_score"):
        await AgentEvaluator.evaluate_eval_set(
            agent_module="agentic_rag.agent",
            eval_set=eval_set,
            eval_config=config,
            num_runs=1,
            print_detailed_results=False,
        )


def test_eval_set_covers_filters_multi_hop_and_refusal() -> None:
    data = json.loads(EVALSET.read_text())
    hops = {
        c["eval_id"]: len(c["conversation"][0]["intermediate_data"]["tool_uses"])
        for c in data["eval_cases"]
    }
    assert hops == {"filtered_lookup": 1, "multi_hop_bridge": 2, "refusal": 1}
    first = data["eval_cases"][0]["conversation"][0]["intermediate_data"]["tool_uses"][0]["args"]
    assert first["product"] == "lumen" and first["version"] == "2.0"
