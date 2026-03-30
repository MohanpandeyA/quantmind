"""Application settings using Pydantic BaseSettings.

All configuration is loaded from environment variables or .env file.
Never hardcode secrets — always use this settings module.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the QuantMind backend.

    Attributes:
        groq_api_key: API key for Groq LLM (free tier).
        groq_model: Groq model name to use.
        news_api_key: API key for NewsAPI (free tier).
        mongodb_uri: MongoDB Atlas connection string.
        chroma_persist_dir: Local directory for ChromaDB persistence.
        fastapi_host: Host to bind the FastAPI server.
        fastapi_port: Port to bind the FastAPI server.
        debug: Enable debug mode.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        openai_api_key: Optional OpenAI key (paid upgrade).
        pinecone_api_key: Optional Pinecone key (paid upgrade).
        polygon_api_key: Optional Polygon.io key (paid upgrade).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Free Tier ---
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(
        default="llama-3.1-70b-versatile", description="Groq model name"
    )
    news_api_key: str = Field(default="", description="NewsAPI key")
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017/quantmind",
        description="MongoDB connection URI",
    )
    chroma_persist_dir: str = Field(
        default="./data/chroma", description="ChromaDB persistence directory"
    )

    # --- Server ---
    fastapi_host: str = Field(default="0.0.0.0", description="FastAPI host")
    fastapi_port: int = Field(default=8000, description="FastAPI port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")

    # --- Alpaca Trading (Free paper trading: https://alpaca.markets) ---
    alpaca_api_key: str = Field(default="", description="Alpaca API key")
    alpaca_secret_key: str = Field(default="", description="Alpaca secret key")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca base URL (paper or live)",
    )

    # --- Paid Upgrades (optional, leave empty) ---
    openai_api_key: str = Field(default="", description="OpenAI API key (paid)")
    pinecone_api_key: str = Field(default="", description="Pinecone API key (paid)")
    pinecone_env: str = Field(default="", description="Pinecone environment (paid)")
    polygon_api_key: str = Field(default="", description="Polygon.io API key (paid)")
    alpha_vantage_api_key: str = Field(
        default="", description="Alpha Vantage API key (paid)"
    )


# Singleton instance — import this everywhere
settings = Settings()
