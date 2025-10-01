"""Rate limiting for AI chat"""
import streamlit as st
from datetime import datetime, timedelta

class RateLimiter:
    """Limit the number of AI queries per user"""
    
    @staticmethod
    def initialize_rate_limit():
        """Initialize rate limiting in session state"""
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        if 'max_queries' not in st.session_state:
            st.session_state.max_queries = 2
    
    @staticmethod
    def can_query() -> bool:
        """Check if user can make another query"""
        RateLimiter.initialize_rate_limit()
        return st.session_state.query_count < st.session_state.max_queries
    
    @staticmethod
    def increment_query():
        """Increment query count"""
        RateLimiter.initialize_rate_limit()
        st.session_state.query_count += 1
    
    @staticmethod
    def get_remaining_queries() -> int:
        """Get number of remaining queries"""
        RateLimiter.initialize_rate_limit()
        return st.session_state.max_queries - st.session_state.query_count
    
    @staticmethod
    def reset_limit():
        """Reset query limit (admin only)"""
        st.session_state.query_count = 0