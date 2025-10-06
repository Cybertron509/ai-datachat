"""
Supabase Client - Database Connection
"""
import os
from supabase import create_client, Client
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Manage Supabase database connections"""
    
    _instance: Optional[Client] = None
    
    @classmethod
    def get_client(cls) -> Client:
        """Get or create Supabase client (singleton)"""
        if cls._instance is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            
            if not url or not key:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
            
            cls._instance = create_client(url, key)
            logger.info("Supabase client initialized")
        
        return cls._instance
    
    @classmethod
    def test_connection(cls) -> bool:
        """Test database connection"""
        try:
            client = cls.get_client()
            # Simple query to test connection
            result = client.table('users').select('count').limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Supabase connection failed: {str(e)}")
            return False