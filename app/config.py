"""
Centralized configuration. Reads from environment variables / .env file.
Keeping all config in one place makes the ops story (which model, which
tracing project, which cost limits) auditable at a glance -- this is a
small but real LLMOps habit worth mentioning in an interview.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings loads .env into OUR Settings object below, but does not
# populate the real OS environment. Some third-party libraries (ragas,
# and earlier we hit this with langchain's ChatOpenAI too) read credentials
# directly from os.environ regardless of our own config object. Calling
# load_dotenv() here ensures both paths see the same values, so we don't
# have to special-case every library that expects os.environ.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider
    openai_api_key: str = ""
    cheap_model: str = "gpt-4o-mini"   # used for research + fact-check (cost-optimized)
    strong_model: str = "gpt-4o"       # used for final synthesis / escalation

    # Search provider (used by the researcher agent)
    tavily_api_key: str = ""

    # Langfuse (tracing / observability)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    enable_tracing: bool = True

    # Agent behavior
    max_search_results: int = 5
    max_agent_iterations: int = 6

    # App
    app_env: str = "development"


settings = Settings()