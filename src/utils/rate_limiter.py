"""
Rate Limiter for AI Chat
Controls API usage to manage costs
"""
import streamlit as st


class RateLimiter:
    """Rate limit AI chat queries per session"""
    
    MAX_QUERIES = 2  # Default for free tier
    
    @staticmethod
    def get_query_count() -> int:
        """Get current query count"""
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        return st.session_state.query_count
    
    @staticmethod
    def can_query() -> bool:
        """Check if user can make another query"""
        return RateLimiter.get_query_count() < RateLimiter.MAX_QUERIES
    
    @staticmethod
    def get_remaining_queries() -> int:
        """Get number of remaining queries"""
        return max(0, RateLimiter.MAX_QUERIES - RateLimiter.get_query_count())
    
    @staticmethod
    def increment_query():
        """Increment query count"""
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        st.session_state.query_count += 1