"""
Authentication with Supabase
"""
import bcrypt
import logging
from datetime import datetime
from .supabase_db import get_db

logger = logging.getLogger(__name__)


class AuthManager:
    """Manage user authentication with Supabase"""
    
    def __init__(self):
        self.db = get_db()
    
    def register_user(self, username: str, password: str, full_name: str, email: str = None) -> bool:
        """Register a new user"""
        if not self.db.is_connected():
            logger.error("Database not connected")
            return False
        
        try:
            # Check if username exists
            existing = self.db.client.table('users').select('id').eq('username', username).execute()
            
            if existing.data:
                logger.warning(f"Username already exists: {username}")
                return False
            
            # Check if email exists (if provided)
            if email:
                existing_email = self.db.client.table('users').select('id').eq('email', email).execute()
                if existing_email.data:
                    logger.warning(f"Email already exists: {email}")
                    return False
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Insert user (email can be empty for now, we'll add it to registration form later)
            user_data = {
                'username': username,
                'email': email or f"{username}@temp.local",  # Temporary email if not provided
                'password_hash': password_hash,
                'full_name': full_name
            }
            
            user_response = self.db.client.table('users').insert(user_data).execute()
            
            if user_response.data:
                user_id = user_response.data[0]['id']
                
                # Create default subscription
                self.db.client.table('subscriptions').insert({
                    'user_id': user_id,
                    'tier': 'free',
                    'status': 'active',
                    'ai_questions_used': 0
                }).execute()
                
                logger.info(f"User registered successfully: {username}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user"""
        if not self.db.is_connected():
            logger.error("Database not connected")
            return False
        
        try:
            user_response = self.db.client.table('users').select(
                'id, password_hash'
            ).eq('username', username).execute()
            
            if not user_response.data:
                logger.warning(f"User not found: {username}")
                return False
            
            user = user_response.data[0]
            
            # Verify password
            if bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                # Update last login
                self.db.client.table('users').update({
                    'last_login': datetime.now().isoformat()
                }).eq('id', user['id']).execute()
                
                logger.info(f"User authenticated: {username}")
                return True
            
            logger.warning(f"Invalid password for user: {username}")
            return False
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def get_user_info(self, username: str) -> dict:
        """Get user information"""
        if not self.db.is_connected():
            return {}
        
        try:
            user_response = self.db.client.table('users').select(
                'id, username, email, full_name, created_at'
            ).eq('username', username).execute()
            
            if user_response.data:
                return user_response.data[0]
            return {}
            
        except Exception as e:
            logger.error(f"Get user info error: {e}")
            return {}