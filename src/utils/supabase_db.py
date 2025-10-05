"""
Supabase database connection
"""
from supabase import create_client, Client
import streamlit as st
import os
import logging

logger = logging.getLogger(__name__)


class SupabaseDB:
    """Supabase database manager"""
    
    def __init__(self):
        """Initialize Supabase client"""
        try:
            url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
            key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))
            
            if not url or not key:
                logger.error("Supabase credentials not found")
                self.client = None
                return
            
            self.client: Client = create_client(url, key)
            logger.info("Successfully connected to Supabase")
            
        except Exception as e:
            logger.error(f"Supabase connection error: {e}")
            self.client = None
    
    def is_connected(self):
        """Check if connected"""
        return self.client is not None


# Singleton instance
_db_instance = None

def get_db():
    """Get or create Supabase instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SupabaseDB()
    return _db_instance