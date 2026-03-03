from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Anthropic API
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_max_tokens: int = 4096

    # Data limits
    max_upload_size_mb: int = 200
    max_rows_for_llm_context: int = 50
    max_columns_for_llm_context: int = 30
    sampling_strategy: str = "stratified"  # "random", "stratified", "head_tail"

    # Validation thresholds
    outlier_z_threshold: float = 3.0
    missing_data_warning_pct: float = 5.0
    missing_data_critical_pct: float = 30.0

    # UI
    streamlit_page_title: str = "Data Intelligence Agent"
    max_chat_history: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    return Settings()
