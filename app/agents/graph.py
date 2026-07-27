import re
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.agents.state import ResearchState
from app.agents.tools import web_search
from app.config import settings

_cheap_llm = None
_strong_llm = None


def get_cheap_llm() -> ChatOpenAI:
    global _cheap_llm
    if _cheap_llm is None:
        _cheap_llm = ChatOpenAI(
            model=settings.cheap_model, temperature=0, api_key=settings.openai_api_key
        )
    return _cheap_llm


def get_strong_llm() -> ChatOpenAI:
    global _strong_llm
    if _strong_llm is None:
        _strong_llm = ChatOpenAI(
            model=settings.strong_model, temperature=0.2, api_key=settings.openai_api_key
        )
    return _strong_llm


def researcher_node(state: ResearchState) -> ResearchState:
    query = state["question"]
    results = web_search.invoke(query)

    sources = re.findall(r"Source:\s*(\S+)", results)

    return {
        **state,
        "search_results": results,
        "sources": sources,
    }


DRAFT_PROMPT = """You are a research assistant. Using ONLY the search results
below, write a concise, factual draft answer to the question. If the search
results don't contain enough information, say so explicitly rather than
guessing.

Question: {question}

Search results:
{search_results}

{revision_notes}

Draft answer:"""


def drafter_node(state: ResearchState) -> ResearchState:
    revision_notes = ""
    if state.get("fact_check_notes"):
        revision_notes = (
            f"\nA fact-checker flagged issues with your previous draft:\n"
            f"{state['fact_check_notes']}\nRevise the draft to address these issues.\n"
        )

    prompt = DRAFT_PROMPT.format(
        question=state["question"],
        search_results=state.get("search_results", "No search results available."),
        revision_notes=revision_notes,
    )
    response = get_cheap_llm().invoke(prompt)

    return {
        **state,
        "draft_answer": response.content,
        "iterations": state.get("iterations", 0) + 1,
    }


FACT_CHECK_PROMPT = """You are a strict fact-checker. Compare the draft answer
against the search results it was based on. Flag any claim in the draft that
is NOT supported by the search results (possible hallucination).

Question: {question}

Search results:
{search_results}

Draft answer:
{draft_answer}

Respond in this exact format:
VERDICT: PASS or FAIL
NOTES: <brief explanation, empty if PASS>"""


def fact_checker_node(state: ResearchState) -> ResearchState:
    prompt = FACT_CHECK_PROMPT.format(
        question=state["question"],
        search_results=state.get("search_results", ""),
        draft_answer=state.get("draft_answer", ""),
    )
    response = get_cheap_llm().invoke(prompt)
    text = response.content

    passed = "VERDICT: PASS" in text.upper()
    notes_match = re.search(r"NOTES:\s*(.*)", text, re.DOTALL)
    notes = notes_match.group(1).strip() if notes_match else ""

    return {
        **state,
        "fact_check_passed": passed,
        "fact_check_notes": notes,
    }


def route_after_fact_check(state: ResearchState) -> Literal["drafter", "summarizer"]:
    """Bounded self-correction: retry once, then move on regardless so the
    graph always terminates."""
    if state.get("fact_check_passed") or state.get("iterations", 0) >= settings.max_agent_iterations:
        return "summarizer"
    return "drafter"


SUMMARY_PROMPT = """You are producing the final answer for a user. Polish the
draft below into a clear, well-organized answer. Add a short "Sources"
section at the end listing the source URLs provided.

Question: {question}

Draft:
{draft_answer}

Sources available: {sources}

Final answer:"""


def summarizer_node(state: ResearchState) -> ResearchState:
    prompt = SUMMARY_PROMPT.format(
        question=state["question"],
        draft_answer=state.get("draft_answer", ""),
        sources=", ".join(state.get("sources", [])) or "None found",
    )
    response = get_strong_llm().invoke(prompt)

    return {
        **state,
        "final_answer": response.content,
        "model_used": settings.strong_model,
    }


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("drafter", drafter_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "drafter")
    graph.add_edge("drafter", "fact_checker")
    graph.add_conditional_edges(
        "fact_checker",
        route_after_fact_check,
        {"drafter": "drafter", "summarizer": "summarizer"},
    )
    graph.add_edge("summarizer", END)

    return graph.compile()


research_graph = build_graph()


def run_research(question: str) -> ResearchState:
    initial_state: ResearchState = {
        "question": question,
        "search_results": None,
        "draft_answer": None,
        "fact_check_notes": None,
        "fact_check_passed": None,
        "final_answer": None,
        "iterations": 0,
        "model_used": None,
        "sources": [],
    }
    return research_graph.invoke(initial_state)
