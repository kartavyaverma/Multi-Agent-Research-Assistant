"""
Centralized configuration. Reads from environment variables / .env file.
Keeping all config in one place makes the ops story (which model, which
tracing project, which cost limits) auditable at a glance -- this is a
small but real LLMOps habit worth mentioning in an interview.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


    openai_api_key: str = ""
    cheap_model: str = "gpt-4o-mini"
    strong_model: str = "gpt-4o"


    tavily_api_key: str = ""


    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    enable_tracing: bool = True


    max_search_results: int = 5
    max_agent_iterations: int = 6


    app_env: str = "development"


settings = Settings()