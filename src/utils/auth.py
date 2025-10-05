"""
Authentication management for AI DataChat
"""
import streamlit as st
import bcrypt
import logging
from .supabase_db import get_db

logger = logging.getLogger(__name__)


class AuthManager:
    """Manage user authentication and registration"""
    
    def __init__(self):
        """Initialize auth manager"""
        self.db = get_db()
        if not self.db:
            logger.error("Failed to initialize database connection")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {str(e)}")
            return False
    
    def register_user(self, username: str, password: str, full_name: str, email: str = None) -> bool:
        """Register a new user"""
        if not self.db:
            logger.error("Database not connected")
            return False
        
        try:
            # Check if username already exists
            result = self.db.table('users').select('username').eq('username', username).execute()
            
            if result.data and len(result.data) > 0:
                st.error("Username already exists")
                return False
            
            # Hash password
            password_hash = self.hash_password(password)
            
            # Insert new user
            user_data = {
                'username': username,
                'password_hash': password_hash,
                'full_name': full_name,
                'email': email or f"{username}@temp.local"
            }
            
            result = self.db.table('users').insert(user_data).execute()
            
            if result.data:
                st.success("Registration successful! Please login.")
                return True
            else:
                st.error("Registration failed")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            st.error(f"Registration error: {str(e)}")
            return False
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user"""
        if not self.db:
            logger.error("Database not connected")
            return False
        
        try:
            # Query user
            result = self.db.table('users').select('*').eq('username', username).execute()
            
            if not result.data or len(result.data) == 0:
                return False
            
            user = result.data[0]
            
            # Verify password
            if self.verify_password(password, user['password_hash']):
                # Store user in session
                st.session_state['user'] = user
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                
                # Update last login
                try:
                    self.db.table('users').update({
                        'last_login': 'now()'
                    }).eq('username', username).execute()
                except Exception as e:
                    logger.warning(f"Could not update last login: {str(e)}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            st.error(f"Login error: {str(e)}")
            return False
    
    def logout(self):
        """Logout current user"""
        st.session_state['authenticated'] = False
        st.session_state['user'] = None
        st.session_state['username'] = None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    def get_current_user(self):
        """Get current authenticated user"""
        return st.session_state.get('user', None)