"""
Configuration settings for Databricks Agent App.
Loads settings from environment variables with sensible defaults.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Lakebase Configuration
    lakebase_instance_name: str = os.getenv("LAKEBASE_INSTANCE_NAME", "")
    lakebase_database_name: str = os.getenv("LAKEBASE_DATABASE_NAME", "")
    lakebase_catalog_name: str = os.getenv("LAKEBASE_CATALOG_NAME", "")

    # Database Connection
    databricks_database_port: int = int(os.getenv("DATABRICKS_DATABASE_PORT", "5432"))
    default_postgres_schema: str = os.getenv("DEFAULT_POSTGRES_SCHEMA", "public")

    # Connection Pool Settings
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    db_pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    db_pool_recycle_interval: int = int(os.getenv("DB_POOL_RECYCLE_INTERVAL", "3600"))

    # Databricks Configuration
    databricks_host: str = os.getenv("DATABRICKS_HOST", "")
    databricks_token: str = os.getenv("DATABRICKS_TOKEN", "")

    # Agent Configuration
    databricks_model_endpoint: str = os.getenv(
        "DATABRICKS_MODEL_ENDPOINT", "databricks-claude-3-5-sonnet"
    )
    agent_system_prompt: str = os.getenv(
        "AGENT_SYSTEM_PROMPT",
        "You are a helpful AI assistant with long-term memory capabilities. "
        "You can remember information from previous conversations.",
    )

    # Token refresh interval (50 minutes to be safe with 1-hour tokens)
    token_refresh_interval: int = 50 * 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
