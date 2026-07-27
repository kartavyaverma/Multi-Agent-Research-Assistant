# Multi-Agent Research Assistant

A multi-agent research pipeline that answers a question by having four
specialized agents divide the work — search, draft, fact-check, and
summarize — instead of relying on a single LLM call. Built with
**LangGraph**, served via **FastAPI**, with a **Streamlit** UI, **Langfuse**
tracing, and **Ragas** evaluation.

```
Question → Researcher → Drafter → Fact-checker → Summarizer → Answer
              (search)   (draft)     ↑    ↓        (polish +
                                retry once           citations)
                                if unsupported
                                claims found
```

## Why this exists

Most RAG demos are a single prompt: retrieve → stuff into context → generate.
This project is deliberately built as a **multi-agent system with
verification and correction as first-class steps**, plus the operational
concerns that matter once "does it answer the question" isn't the only bar:

- **Correctness**: a fact-checker agent compares the draft against retrieved
  sources and can send it back for one revision before it reaches the user.
- **Cost**: cheap model (`gpt-4o-mini`) handles research and fact-checking;
  the expensive model (`gpt-4o`) is only used once, for final synthesis.
- **Observability**: every agent step is traced (input, output, latency,
  cost) via Langfuse — not just "it returned 200 OK."
- **Measured quality**: Ragas scores every answer for faithfulness (is it
  grounded in the retrieved sources?) and relevancy, so quality can be
  tracked over time instead of eyeballed.

## Results

Evaluated on a held-out set of research questions using
[Ragas](https://github.com/explodinggradients/ragas):

| Metric | Score |
|---|---|
| Faithfulness | **0.99** |
| Answer Relevancy | **0.98** |

(Faithfulness measures whether every claim in the final answer is
traceable to the retrieved search results — i.e. how much the pipeline
avoids hallucinating.)

## Architecture

| Agent | Model | Job |
|---|---|---|
| **Researcher** | — | Searches the web via [Tavily](https://tavily.com) for current, relevant sources |
| **Drafter** | `gpt-4o-mini` | Writes an answer grounded *only* in the retrieved results |
| **Fact-checker** | `gpt-4o-mini` | Flags unsupported claims; sends the draft back for one revision if needed |
| **Summarizer** | `gpt-4o` | Polishes the final answer and adds citations |

The retry loop is **bounded** (`MAX_AGENT_ITERATIONS` in config) — if the
fact-checker keeps rejecting drafts, the graph still terminates and returns
the best available answer, rather than looping forever.

## Tech stack

- **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) — explicit state graph with conditional routing, not a linear chain
- **Search**: [Tavily](https://tavily.com) — API built for LLM agents (structured results, no scraping fragility)
- **Serving**: FastAPI
- **UI**: Streamlit (thin client, calls the FastAPI backend over HTTP)
- **Observability**: [Langfuse](https://langfuse.com) — per-agent-step tracing
- **Evaluation**: [Ragas](https://github.com/explodinggradients/ragas) — faithfulness & answer relevancy scoring
- **Environment**: [uv](https://docs.astral.sh/uv/) — reproducible dependency management
- **CI/CD**: GitHub Actions (tests + Docker build on every push)

## Project structure

```
app/
  agents/
    state.py       # shared state schema passed between agents
    tools.py       # Tavily web search tool
    graph.py        # LangGraph orchestration — the core of the pipeline
  config.py        # env-based settings
  tracing.py       # Langfuse integration
  eval.py           # Ragas evaluation script
  main.py           # FastAPI app
streamlit_app.py    # UI — calls the FastAPI backend, no duplicated logic
tests/               # unit + API tests (mocked, no API key required)
Dockerfile           # uv-based container build
.github/workflows/ci.yml
context.md           # design rationale + interview Q&A
```

## Running it locally

Requires an [OpenAI API key](https://platform.openai.com/api-keys) and a
free [Tavily API key](https://tavily.com).

```bash
# 1. Install uv, if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# then edit .env: set OPENAI_API_KEY and TAVILY_API_KEY

# 4. Run the tests (no API key needed — LLM calls are mocked)
uv run pytest tests/ -v

# 5. Start the backend
uv run uvicorn app.main:app --reload

# 6. In a second terminal, start the UI
uv run streamlit run streamlit_app.py
```

Open `http://localhost:8501` for the UI, or `http://localhost:8000/docs`
for the raw API (Swagger).

### Optional: tracing & evaluation

```bash
# Enable Langfuse tracing: add LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
# to .env and set ENABLE_TRACING=true, then just run requests as normal —
# traces appear in your Langfuse project automatically.

# Run the Ragas evaluation suite
uv run python -m app.eval
```

### Docker

```bash
docker build -t research-assistant .
docker run -p 8000:8000 --env-file .env research-assistant
```
