# Multi-Agent Research Assistant — Project Context

## WHAT

A multi-agent system that answers a research question by running it through
four cooperating agents, orchestrated as a graph rather than a single
prompt:

1. **Researcher** — searches the web for current information.
2. **Drafter** — writes an answer grounded only in what the researcher found.
3. **Fact-checker** — checks the draft against the search results, flags
   unsupported claims, and can send it back for one revision.
4. **Summarizer** — polishes the final answer and adds citations.

It's served as a FastAPI HTTP API, containerized with Docker, traced with
Langfuse, and evaluated with Ragas.

## WHY

Three things this project is deliberately designed to demonstrate:

1. **Multi-agent orchestration, not just RAG.** A single-prompt RAG chatbot
   answers a question in one LLM call. This project shows I can decompose a
   task across agents with explicit handoffs, and — critically — recover
   from a bad output (the fact-checker/drafter retry loop) instead of just
   trusting the first answer.

2. **Cost-aware model routing.** Cheap/fast model (`gpt-4o-mini`) handles
   research and fact-checking; the expensive model (`gpt-4o`) is only
   invoked once, for final synthesis. This mirrors how production systems
   actually control LLM spend — not every step needs the strongest model.

3. **LLMOps maturity over generic DevOps.** Instead of bolting on
   Kubernetes/Jenkins-style tooling, the operational layer here is
   LLM-specific: Langfuse traces every agent step (latency, input/output,
   cost), and Ragas scores faithfulness/answer-relevancy so quality
   regressions are caught before shipping — not generic CI for its own sake.

## HOW

- **LangGraph** defines the pipeline as a `StateGraph` with explicit nodes
  and a conditional edge (`route_after_fact_check`) that implements the
  bounded retry loop. Bounded matters: without an iteration cap, two
  disagreeing agents (drafter vs. fact-checker) could loop forever — a real
  failure mode in production agentic systems.
- **State** is a single `TypedDict` (`ResearchState`) passed between nodes,
  rather than raw chat history. This makes every agent's input/output
  independently inspectable and testable (see `tests/test_graph.py`, which
  tests routing logic and source-extraction without calling any LLM).
- **Tracing** wraps each `/research` request in a Langfuse trace, with a
  span per agent step, so a slow or expensive request can be diagnosed
  node-by-node instead of as one opaque black box.
- **Evaluation** (`app/eval.py`) runs a fixed set of questions through the
  full pipeline and scores the outputs with Ragas metrics, saving results
  to CSV for tracking over time / across prompt changes.
- **Deployment**: Dockerfile + a GitHub Actions workflow that runs tests and
  builds the image on every push — deliberately lightweight CI, since this
  is an AI engineering project, not a platform engineering one.
- **Environment management: uv.** `pyproject.toml` + `uv.lock` replace
  `requirements.txt` + manual `venv`. `uv sync` installs the exact locked
  versions (including transitive dependencies) in about a second, and
  `uv run <command>` executes inside the project's virtual environment
  without a manual `source .venv/bin/activate` step. The Dockerfile and CI
  workflow both use `uv` too, so local dev, CI, and the container all
  resolve dependencies from the same lockfile — no "works on my machine."

---

# Interview Prep

## "Why did you build this?"
To show I can go beyond a single-call RAG chatbot and build a system where
multiple agents divide a task, with a way to catch and correct bad output
before it reaches the user — and to do it with realistic constraints
(cost, latency, observability) rather than just a notebook demo.

## "Walk me through the architecture."
Researcher pulls live web results → Drafter writes a grounded answer using
only those results → Fact-checker compares the draft against the search
results and either passes it or sends it back with notes → Summarizer
polishes the (validated) draft and adds citations, using a stronger model
since it's the one output the user actually sees. The whole thing is a
LangGraph `StateGraph`, not a chain of separate scripts, so the control flow
(including the retry loop) is explicit and visualizable.

## "Why LangGraph instead of a simple chain or plain function calls?"
A linear chain can't express "go back and retry" cleanly. LangGraph lets me
define that as a conditional edge on shared state, which is closer to how
these systems actually behave in production — agents disagree, and you need
an explicit, bounded way to reconcile that instead of hoping the first pass
is right.

## "How do you prevent infinite loops between the drafter and fact-checker?"
`iterations` is incremented every drafter pass and checked in
`route_after_fact_check`; once it hits `max_agent_iterations` the graph
routes to the summarizer regardless of fact-check verdict. This guarantees
termination — a bounded retry with a fallback path, rather than a hard
requirement that agents eventually agree.

## "How do you know the fact-checker is actually catching hallucinations, not just rubber-stamping?"
Two ways: (1) the fact-checker's prompt explicitly requires it to compare
each claim against the retrieved search results, not just judge plausibility;
(2) independently, the Ragas `faithfulness` metric scores the *final* answer
against the retrieved context after the fact is all said and done, so I have
an evaluation signal that isn't just "trust the fact-checker agent."

## "Why gpt-4o-mini for most steps and gpt-4o only at the end?"
Cost and latency. Research and fact-checking are largely
extraction/verification tasks a smaller model handles well; the final
summary is the one output quality actually matters most for user-facing
polish, so that's the one step worth paying for a stronger model. In a real
system, I'd back this decision with actual cost/quality data from the eval
pipeline rather than assuming it.

## "Tell me about a bug you hit and fixed while building this."
Two good examples surfaced building this, both about credential handling —
worth having ready since it's a common interview thread ("how do you debug
a production issue," "tell me about a subtle bug"):

1. **Import-time client construction.** `ChatOpenAI` clients were originally
   built at module import time. That meant simply importing the module —
   which `pytest` does when collecting tests — raised
   `OpenAIError: api_key must be set`, even for tests that fully mock the
   LLM call and never need a real key. Fixed with lazy getters
   (`get_cheap_llm()` / `get_strong_llm()`) that only construct the client
   on first real use.

2. **Two independent config-loading paths.** `pydantic-settings` loads
   `.env` into our own `Settings` object (`settings.openai_api_key`) — but
   `ChatOpenAI(...)` doesn't read that object at all; by default it does its
   own independent lookup of `OPENAI_API_KEY` from the real OS environment.
   Since nothing ever copied the value from `Settings` into `os.environ`,
   the key was correctly loaded into our app's config but never reached
   LangChain's client, producing the same "api_key must be set" error even
   with a perfectly valid `.env` file. Fixed by explicitly passing
   `api_key=settings.openai_api_key` into both `ChatOpenAI(...)` calls,
   so there's a single source of truth for the credential instead of two
   loading mechanisms that silently didn't talk to each other. This is a
   good example of a bug that *looks* like a config/environment mistake but
   is actually an integration bug between two libraries that each manage
   config their own way — worth explicitly checking library defaults
   instead of assuming "loaded into my settings object" means "available
   to every client that needs it."

3. **DuckDuckGo scraping got rate-limited in real use.** The researcher
   agent originally used `duckduckgo_search`, which has no official API —
   it scrapes DuckDuckGo's HTML. In practice this returned
   `202 Ratelimit` errors after a handful of requests, so the pipeline
   silently fell back to the LLM's own training knowledge with an empty
   `sources` list — technically a "200 OK" response, but no longer actually
   doing research. Fixed by switching to **Tavily**, a search API built
   specifically for LLM agents (structured JSON, no scraping fragility,
   generous free tier). Good interview point: a response can look
   successful (valid JSON, 200 status) while silently failing on the thing
   that actually mattered — worth checking not just "did it return 200"
   but "did the tool it depended on actually run."

## "What would break this in production, and how would you fix it?"
- **Search API reliability**: even Tavily (or any third-party API) can have
  an outage or hit its rate limit under real load. I'd add retries with
  backoff and a circuit breaker, and — importantly — make the researcher
  node surface a clear failure/low-confidence signal (not just an empty
  `sources` list that a downstream agent quietly writes around).
- **Fact-checker false negatives**: an LLM fact-checker can itself
  hallucinate a "PASS." I'd add periodic human-in-the-loop spot checks and
  track faithfulness scores over time as a leading indicator of drift.
- **Cost blowup on the retry loop**: if `max_agent_iterations` were set too
  high, a consistently-failing fact-check could get expensive. I'd add a
  hard token/cost budget per request, not just an iteration count.

## "How would you extend this?"
- Add a **router agent** at the front that decides whether a question even
  needs web search (some questions are answerable from the model's own
  knowledge), saving a search call and reducing latency.
- Add **streaming** so the user sees the researcher/drafter/summarizer
  stages progressively instead of waiting for the full pipeline.
- Add **caching** on the search/draft steps so repeated or similar
  questions don't re-run the full pipeline.

## "What's the difference between this and a RAG chatbot?"
RAG typically means: retrieve → stuff into one prompt → generate. This is
that idea extended with **verification and correction as first-class
steps**, and with the retrieval and generation split across agents with
different responsibilities and different cost profiles — closer to how a
production system would actually be built once "does it answer the
question" isn't enough and "is it correct, and can I afford to run it" also
matter.

## Common general LLM/agent interview questions to be ready for
- What's the difference between LangChain and LangGraph, and when would you
  use each?
- How do you handle prompt injection in a tool-using agent (e.g., a
  malicious webpage in the search results instructing the model to ignore
  its instructions)?
- How do you evaluate an LLM system without ground-truth labels?
- What's the tradeoff between a single large prompt and a multi-agent
  decomposition?
- How would you reduce latency in a multi-step agent pipeline?
- How do you version and test prompts as they change over time?
- What's your approach to handling an LLM API outage or rate limit mid-request?
