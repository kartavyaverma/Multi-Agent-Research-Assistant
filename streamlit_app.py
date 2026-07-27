import time

import requests
import streamlit as st

API_URL = "http://localhost:8000/research"
HEALTH_URL = "http://localhost:8000/health"

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔎", layout="centered")

st.title("🔎 Multi-Agent Research Assistant")
st.caption("Researcher → Drafter → Fact-checker → Summarizer, backed by LangGraph + FastAPI")

try:
    health = requests.get(HEALTH_URL, timeout=3)
    backend_ok = health.status_code == 200
except requests.exceptions.RequestException:
    backend_ok = False

if not backend_ok:
    st.error(
        "Can't reach the backend at http://localhost:8000. "
        "Start it first with: `uv run uvicorn app.main:app --reload`"
    )
    st.stop()

question = st.text_input(
    "Ask a research question",
    placeholder="e.g. What are the health effects of intermittent fasting?",
)

run_button = st.button("Research", type="primary", disabled=not question.strip())

if run_button and question.strip():
    with st.spinner("Researching... (typically 10-20s: search → draft → fact-check → summarize)"):
        start = time.time()
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=120)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
            st.stop()
        elapsed = time.time() - start

    st.success(f"Done in {elapsed:.1f}s")

    st.subheader("Answer")
    st.markdown(data["final_answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Fact-check passed", "✅ Yes" if data["fact_check_passed"] else "❌ No")
    col2.metric("Iterations", data["iterations"])
    col3.metric("Model used", data["model_used"])

    if data.get("sources"):
        st.subheader("Sources")
        for src in data["sources"]:
            st.markdown(f"- [{src}]({src})")
    else:
        st.warning("No sources were returned for this answer.")

    with st.expander("Raw response JSON"):
        st.json(data)
