"""
Supabase database connection utilities
"""
import os
import streamlit as st
from supabase import create_client, Client
from typing import Optional


class SupabaseDB:
    """Singleton class for Supabase database connection"""
    
    _instance: Optional[Client] = None
    
    @classmethod
    def get_client(cls) -> Optional[Client]:
        """Get or create Supabase client"""
        if cls._instance is None:
            try:
                # Check environment variables first (for Render), then Streamlit secrets (for Streamlit Cloud)
                url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
                key = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
                
                if not url or not key:
                    print("ERROR: Supabase credentials not found in environment variables or secrets")
                    return None
                
                cls._instance = create_client(url, key)
                print("Successfully connected to Supabase")
                
            except Exception as e:
                print(f"Supabase connection error: {str(e)}")
                return None
        
        return cls._instance


def get_db() -> Optional[Client]:
    """Get Supabase client instance"""
    return SupabaseDB.get_client()