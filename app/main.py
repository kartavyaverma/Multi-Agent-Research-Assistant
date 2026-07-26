"""
FastAPI entrypoint. Exposes the multi-agent research assistant as an HTTP
API, with tracing wrapped around every request.
"""

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agents.graph import run_research
from app.tracing import trace_research_run, log_step

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="Researcher -> Drafter -> Fact-checker -> Summarizer, with cost-aware model routing and Langfuse tracing.",
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What are the health effects of intermittent fasting?"])


class ResearchResponse(BaseModel):
    question: str
    final_answer: str
    sources: list[str]
    fact_check_passed: bool
    iterations: int
    model_used: str
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(payload: ResearchRequest):
    start = time.perf_counter()

    with trace_research_run(payload.question) as trace:
        try:
            result = run_research(payload.question)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {exc}") from exc

        log_step(trace, "researcher", payload.question, result.get("search_results"))
        log_step(trace, "drafter", result.get("search_results"), result.get("draft_answer"))
        log_step(trace, "fact_checker", result.get("draft_answer"), result.get("fact_check_notes"))
        log_step(
            trace, "summarizer", result.get("draft_answer"), result.get("final_answer"),
            model=result.get("model_used"),
        )

    latency = time.perf_counter() - start

    if not result.get("final_answer"):
        raise HTTPException(status_code=500, detail="No final answer produced.")

    return ResearchResponse(
        question=payload.question,
        final_answer=result["final_answer"],
        sources=result.get("sources", []),
        fact_check_passed=bool(result.get("fact_check_passed")),
        iterations=result.get("iterations", 0),
        model_used=result.get("model_used", "unknown"),
        latency_seconds=round(latency, 2),
    )
