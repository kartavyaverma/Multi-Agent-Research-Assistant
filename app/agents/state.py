"""
Shared state that flows through the LangGraph graph. Every node reads from
and writes to this single object, which is what makes the handoff between
agents explicit and inspectable (and traceable in Langfuse).
"""

from typing import TypedDict, List, Optional


class ResearchState(TypedDict):
    question: str                       # original user question
    search_results: Optional[str]       # raw output from the researcher agent
    draft_answer: Optional[str]         # answer produced before fact-checking
    fact_check_notes: Optional[str]     # issues found by the fact-checker agent
    fact_check_passed: Optional[bool]   # whether the draft cleared fact-checking
    final_answer: Optional[str]         # summarizer's polished, cited answer
    iterations: int                     # loop guard so re-drafting can't run forever
    model_used: Optional[str]           # which model actually answered (for cost tracking)
    sources: List[str]                  # extracted source URLs for citation
