"""
Unit tests. Node-level functions are tested independently of the LLM by
monkeypatching the model calls, so tests run fast and don't need an API key.
"""

from unittest.mock import patch, MagicMock

from app.agents.state import ResearchState
from app.agents.graph import route_after_fact_check, researcher_node


def make_state(**overrides) -> ResearchState:
    base: ResearchState = {
        "question": "test question",
        "search_results": None,
        "draft_answer": None,
        "fact_check_notes": None,
        "fact_check_passed": None,
        "final_answer": None,
        "iterations": 0,
        "model_used": None,
        "sources": [],
    }
    base.update(overrides)
    return base


def test_route_after_fact_check_passes_goes_to_summarizer():
    state = make_state(fact_check_passed=True, iterations=1)
    assert route_after_fact_check(state) == "summarizer"


def test_route_after_fact_check_fails_retries_drafter():
    state = make_state(fact_check_passed=False, iterations=1)
    assert route_after_fact_check(state) == "drafter"


def test_route_after_fact_check_stops_at_max_iterations():
    state = make_state(fact_check_passed=False, iterations=6)
    assert route_after_fact_check(state) == "summarizer"


@patch("app.agents.graph.web_search")
def test_researcher_node_extracts_sources(mock_search):
    mock_search.invoke.return_value = (
        "[1] Title one\nBody\nSource: https://example.com/a\n\n"
        "[2] Title two\nBody\nSource: https://example.com/b"
    )
    state = make_state(question="anything")
    result = researcher_node(state)

    assert result["sources"] == ["https://example.com/a", "https://example.com/b"]
    assert result["search_results"] is not None
