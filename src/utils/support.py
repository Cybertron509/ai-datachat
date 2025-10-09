"""
Support ticket system
"""
from datetime import datetime
import logging
from .supabase_db import get_db

logger = logging.getLogger(__name__)


class SupportSystem:
    """Manage support tickets"""
    
    def __init__(self):
        self.db = get_db()
    
    def create_ticket(self, username: str, subject: str, message: str, 
                     priority: str = 'medium') -> int:
        """Create a new support ticket"""
        if not self.db.is_connected():
            logger.error("Database not connected")
            return 0
        
        try:
            # Get user info
            user_response = self.db.client.table('users').select('id, email').eq('username', username).execute()
            
            if not user_response.data:
                logger.error(f"User not found: {username}")
                return 0
            
            user = user_response.data[0]
            user_id = user['id']
            
            # Create ticket
            ticket_response = self.db.client.table('support_tickets').insert({
                'user_id': user_id,
                'subject': subject,
                'message': message,
                'status': 'open',
                'priority': priority
            }).execute()
            
            if ticket_response.data:
                ticket_id = ticket_response.data[0]['id']
                logger.info(f"Support ticket created: #{ticket_id} by {username}")
                return ticket_id
            
            return 0
            
        except Exception as e:
            logger.error(f"Create ticket error: {e}")
            return 0
    
    def get_user_tickets(self, username: str):
        """Get all tickets for a user"""
        if not self.db.is_connected():
            return []
        
        try:
            # Get user ID first
            user_response = self.db.client.table('users').select('id').eq('username', username).execute()
            
            if not user_response.data:
                return []
            
            user_id = user_response.data[0]['id']
            
            # Get tickets
            tickets_response = self.db.client.table('support_tickets').select('*').eq(
                'user_id', user_id
            ).order('created_at', desc=True).execute()
            
            return tickets_response.data or []
            
        except Exception as e:
            logger.error(f"Get user tickets error: {e}")
            return []