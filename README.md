# Multi-Agent Research Assistant

A multi-agent research pipeline built with **LangGraph**, served via **FastAPI**,
with **Langfuse** tracing and **Ragas** evaluation. Environment managed with
**uv**.

## Architecture

```
User question
     │
     ▼
┌────────────┐     ┌──────────┐     ┌──────────────┐     ┌────────────┐
│ Researcher │ ──▶ │ Drafter  │ ──▶ │ Fact-checker │ ──▶ │ Summarizer │ ──▶ Final answer
│ (web search)│    │ (gpt-4o- │     │ (gpt-4o-mini)│     │ (gpt-4o)   │
└────────────┘     │  mini)   │     └──────┬───────┘     └────────────┘
                    └──────────┘            │
                         ▲                  │ FAIL (retry once)
                         └──────────────────┘
```

- **Researcher**: pulls live web search results via DuckDuckGo.
- **Drafter**: writes an answer grounded only in those results (cheap model).
- **Fact-checker**: checks the draft for unsupported claims; can send it back
  for one revision (bounded loop, so it always terminates).
- **Summarizer**: polishes the final answer and adds citations, escalating to
  a stronger model only for this last step (cost-aware routing).

Every request is wrapped in a Langfuse trace so each agent's input/output,
latency, and cost is inspectable per run.

---

## Step-by-step setup (uv)

All commands below were actually run against this exact codebase to confirm
they work — expected output is included so you know what a correct run
looks like.

### 0. Prerequisite: install uv (skip if already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```bash
uv --version
```
**Expected output:**
```
uv 0.11.7 (x86_64-unknown-linux-gnu)
```
(exact version string may differ — any recent uv version is fine)

### 1. Open the project folder in Antigravity / your IDE

Unzip `multi-agent-research-assistant.zip` and open that folder as the
workspace root. `pyproject.toml` at the root is what makes it a recognizable
Python project.

### 2. Create the virtual environment

```bash
uv venv
```
**Expected output:**
```
Using CPython 3.12.3 interpreter at: /usr/bin/python3
Creating virtual environment at: .venv
```
This creates `.venv/` in the project folder. You do **not** need to manually
activate it for the commands below — `uv run` handles that automatically.
(If you want to activate it in your terminal anyway: `source .venv/bin/activate`
on Mac/Linux, `.venv\Scripts\activate` on Windows.)

### 3. Install dependencies

```bash
uv sync
```
**Expected output (abridged):**
```
Resolved 90 packages in 1.2s
Prepared 90 packages in ...
Installed 90 packages in ...
 + fastapi==0.115.6
 + langchain==0.3.14
 + langgraph==0.2.60
 + langfuse==2.57.0
 + ragas==0.2.10
 + uvicorn==0.34.0
 ... (plus their sub-dependencies)
```
This reads `pyproject.toml` + `uv.lock` and installs the exact pinned
versions into `.venv/` — no separate `pip install -r requirements.txt` step
needed, and no "works on my machine" drift since `uv.lock` pins every
transitive dependency too.

### 4. Set your environment variables

```bash
cp .env.example .env
```
Then open `.env` and set at minimum:
```
OPENAI_API_KEY=sk-your-real-key
```
Langfuse keys are optional — if left blank, tracing silently no-ops (see
`app/tracing.py`), so the app still runs without a Langfuse account.

### 5. Run the tests (no API key required)

```bash
uv run pytest tests/ -v
```
**Expected output:**
```
collecting ... collected 6 items

tests/test_api.py::test_health PASSED                                    [ 16%]
tests/test_api.py::test_research_rejects_short_question PASSED           [ 33%]
tests/test_graph.py::test_route_after_fact_check_passes_goes_to_summarizer PASSED [ 50%]
tests/test_graph.py::test_route_after_fact_check_fails_retries_drafter PASSED [ 66%]
tests/test_graph.py::test_route_after_fact_check_stops_at_max_iterations PASSED [ 83%]
tests/test_graph.py::test_researcher_node_extracts_sources PASSED        [100%]

============================== 6 passed in 1.81s ===============================
```
These tests mock the LLM calls entirely, which is why they pass without an
`OPENAI_API_KEY` set — that's intentional (see "lazy LLM initialization" in
`context.md`).

### 6. Run the server locally

```bash
uv run uvicorn app.main:app --reload
```
**Expected output:**
```
INFO:     Will watch for changes in these directories: ['.../multi-agent-research-assistant']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 7. Verify it's alive

In a second terminal:
```bash
curl http://localhost:8000/health
```
**Expected output:**
```json
{"status":"ok"}
```

Interactive API docs (open in browser): `http://localhost:8000/docs`

### 8. Call the actual research endpoint (requires a valid OPENAI_API_KEY)

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the health effects of intermittent fasting?"}'
```
**Expected output shape** (content will vary by run):
```json
{
  "question": "What are the health effects of intermittent fasting?",
  "final_answer": "...polished answer with a Sources section...",
  "sources": ["https://...", "https://..."],
  "fact_check_passed": true,
  "iterations": 1,
  "model_used": "gpt-4o",
  "latency_seconds": 8.42
}
```
`iterations` will be `2` if the fact-checker sent the draft back once before
passing it.

---

## Run with Docker (uv-based image)

```bash
docker build -t research-assistant .
docker run -p 8000:8000 --env-file .env research-assistant
```
The Dockerfile installs `uv` inside the image and runs `uv sync --frozen`
against the committed `uv.lock`, so the container gets the exact same
dependency versions as your local `.venv` — no separate `requirements.txt`
to keep in sync.

## Run evaluation

```bash
uv run python -m app.eval
```
Produces `eval_results.csv` with faithfulness and answer-relevancy scores
per question — run this after any prompt or model change to catch quality
regressions before they ship. Requires a valid `OPENAI_API_KEY`.

## Project layout

```
pyproject.toml    # dependencies (uv-managed)
uv.lock            # pinned, reproducible dependency versions
app/
  agents/
    state.py       # shared state schema passed between agents
    tools.py       # web search tool
    graph.py       # LangGraph orchestration (the core of the project)
  config.py        # env-based settings
  tracing.py       # Langfuse integration
  eval.py          # Ragas evaluation script
  main.py          # FastAPI app
tests/             # unit + API tests (no API key required)
Dockerfile         # uv-based container build
.github/workflows/ci.yml   # uv-based CI
context.md         # why/what/how + interview prep
```
