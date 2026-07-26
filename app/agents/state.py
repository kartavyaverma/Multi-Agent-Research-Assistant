"""
Shared state that flows through the LangGraph graph. Every node reads from
and writes to this single object, which is what makes the handoff between
agents explicit and inspectable (and traceable in Langfuse).
"""

from typing import TypedDict, List, Optional


class ResearchState(TypedDict):
    question: str
    search_results: Optional[str]
    draft_answer: Optional[str]
    fact_check_notes: Optional[str]
    fact_check_passed: Optional[bool]
    final_answer: Optional[str]
    iterations: int
    model_used: Optional[str]
    sources: List[str]
