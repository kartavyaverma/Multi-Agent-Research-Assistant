from contextlib import contextmanager

from langfuse import Langfuse

from app.config import settings

_langfuse_client = None
if settings.enable_tracing and settings.langfuse_public_key:
    _langfuse_client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


@contextmanager
def trace_research_run(question: str):
    """Context manager that opens a Langfuse trace for one end-to-end
    research request and always closes/flushes it, even on error."""
    if _langfuse_client is None:
        yield None
        return

    trace = _langfuse_client.trace(name="research-assistant-run", input={"question": question})
    try:
        yield trace
        trace.update(output="completed")
    except Exception as exc:  # noqa: BLE001
        trace.update(output=f"error: {exc}")
        raise
    finally:
        _langfuse_client.flush()


def log_step(trace, name: str, input_data, output_data, model: str | None = None):
    """Log a single agent step (span) under the current trace, if tracing
    is enabled."""
    if trace is None:
        return
    trace.span(name=name, input=input_data, output=output_data, metadata={"model": model} if model else None)
