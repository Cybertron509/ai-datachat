"""
Configuration settings for AI DataChat
"""
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # App Configuration
    app_name: str = os.getenv("APP_NAME", "AI DataChat")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Data Processing
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "2000"))
    max_rows_display: int = int(os.getenv("MAX_ROWS_DISPLAY", "1000"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "5000"))


settings = Settings()