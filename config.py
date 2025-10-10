"""
Configuration for AI DataChat
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # App settings
    APP_NAME = "AI DataChat"
    VERSION = "1.0.0"
    
    # Azure OpenAI settings
    USE_AZURE_OPENAI = os.getenv('USE_AZURE_OPENAI', 'false').lower() == 'true'
    AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-35-turbo')
    AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
    
    # Regular OpenAI (backup)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Stripe settings
    STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
    STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY')
    
    # Other settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')

settings = Settings()