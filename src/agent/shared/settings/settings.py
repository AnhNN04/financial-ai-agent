# src/stock_assistant/shared/settings/settings.py
import os
import yaml
from typing import Optional
from dotenv import find_dotenv, load_dotenv
# Tìm và tải file .env từ thư mục gốc của dự án
load_dotenv(find_dotenv())
from pydantic_settings import BaseSettings, SettingsConfigDict
from ...shared.logging.logger import Logger
logger = Logger.get_logger(__name__)


class LLMSettings(BaseSettings):
    # General LLM settings
    default_provider: str = "openai"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 60
    max_retries: int = 3
    
    # OpenAI specific settings
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 2000
    openai_top_p: float = 1.0
    openai_frequency_penalty: float = 0.0
    openai_presence_penalty: float = 0.0
    openai_timeout: int = 60
    openai_max_retries: int = 3
    
    # Gemini specific settings
    gemini_model: str = "gemini-1.5-pro"
    gemini_temperature: float = 0.1
    gemini_max_tokens: int = 2000
    gemini_top_p: float = 1.0
    gemini_top_k: int = 40
    gemini_timeout: int = 60
    gemini_max_retries: int = 3
    
    # LangChain settings
    verbose: bool = False
    streaming: bool = False
    enable_callbacks: bool = True
    
    # Safety and content filtering
    enable_content_filter: bool = True
    
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file_encoding='utf-8',
        case_sensitive=False
    )

class QdrantSettings(BaseSettings):
    url: str
    api_key: str
    collection_name: str = "milano-agent-qdrant"
    vector_size: int = 1024
    distance: str = "Cosine"

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        env_file_encoding='utf-8',
        case_sensitive=False
    )

class S3Settings(BaseSettings):
    bucket_name: str
    documents_prefix: str = "ai-agent/"
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str

    model_config = SettingsConfigDict(
        env_prefix="S3_",
        env_file_encoding='utf-8',
        case_sensitive=False
    )

class AppSettings(BaseSettings):
    name: str = "Stock Assistant"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    environment: str = "development"
    
    # API Keys
    openai_api_key: str
    tavily_api_key: str
    gemini_api_key: str

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file_encoding='utf-8',
        case_sensitive=False
    )

class EmbeddingsSettings(BaseSettings):
    # General settings
    model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Cohere v3 specific settings
    cohere_model_id: str = "cohere.embed-multilingual-v3"
    cohere_input_type: str = "search_document"
    cohere_embedding_type: str = "float"
    cohere_max_batch_size: int = 96

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDINGS_",
        env_file_encoding='utf-8',
        case_sensitive=False
    )

class Settings:
    def __init__(self, config_file: str = "settings.yaml"):
        try:
            self.app = AppSettings()
            self.llm = LLMSettings()
            self.qdrant = QdrantSettings()
            self.embeddings = EmbeddingsSettings()
            self.s3 = S3Settings()
            
            # Validate required environment variables
            self._validate_required_configs()
            
            # Load additional config from YAML if exists
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        yaml_config = yaml.safe_load(f)
                        if yaml_config:
                            self._update_from_yaml(yaml_config)
                        else:
                            logger.warning(f"YAML config file {config_file} is empty")
                except Exception as e:
                    logger.error(f"Failed to load YAML config from {config_file}: {str(e)}")
                    raise
        except Exception as e:
            logger.error(f"Failed to initialize settings: {str(e)}")
            raise

    def _validate_required_configs(self):
        """Validate required environment variables."""
        required_fields = [
            (self.app.openai_api_key, "APP_OPENAI_API_KEY"),
            (self.app.tavily_api_key, "APP_TAVILY_API_KEY"),
            (self.app.gemini_api_key, "APP_GEMINI_API_KEY"),
            (self.qdrant.url, "QDRANT_URL"),
            (self.qdrant.api_key, "QDRANT_API_KEY"),
            (self.s3.bucket_name, "S3_BUCKET_NAME"),
            (self.s3.aws_access_key_id, "S3_AWS_ACCESS_KEY_ID"),
            (self.s3.aws_secret_access_key, "S3_AWS_SECRET_ACCESS_KEY"),
            (self.s3.aws_region, "S3_AWS_REGION")
        ]
        for value, field_name in required_fields:
            if not value:
                logger.error(f"Missing required environment variable: {field_name}")
                raise ValueError(f"Missing required environment variable: {field_name}")

    def _update_from_yaml(self, config: dict):
        """Update settings from YAML configuration."""
        for section, values in config.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                    else:
                        logger.warning(f"Ignoring unknown config key: {section}.{key}")

settings = Settings()