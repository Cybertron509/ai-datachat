"""
Subscription management with Supabase
VERSION: 5.0 - Supabase integration
"""
from datetime import datetime, timedelta
from typing import Dict
import logging
from .supabase_db import get_db

logger = logging.getLogger(__name__)


class SubscriptionManager:
    """Manage user subscriptions with Supabase"""
    
    TIERS = {
        'free': {
            'name': 'Free',
            'price': 0,
            'features': [
                'Upload data (CSV, Excel, JSON)',
                'Basic visualizations',
                'Statistical analysis',
                'Data filtering and cleaning',
                '2 AI chat questions (lifetime)'
            ],
            'limits': {
                'ai_questions': 2,
                'forecasting': False,
                'scenarios': False,
                'trust_score': False,
                'reports': False,
                'file_size_mb': 100
            }
        },
        'pro': {
            'name': 'Pro',
            'price': 24.99,
            'features': [
                'Everything in Free, plus:',
                'Unlimited AI chat',
                'Time-series forecasting',
                'Scenario simulation',
                'Data quality trust score',
                'Narrative report generation',
                'Priority support',
                'Larger file uploads (2GB)'
            ],
            'limits': {
                'ai_questions': 999999,
                'forecasting': True,
                'scenarios': True,
                'trust_score': True,
                'reports': True,
                'file_size_mb': 2048
            }
        }
    }
    
    def __init__(self):
        self.db = get_db()
    
    def get_user_subscription(self, username: str) -> Dict:
        """Get user subscription from Supabase"""
        if not self.db:
            logger.warning("Database not connected, returning default subscription")
            return {'tier': 'free', 'status': 'active', 'ai_questions_used': 0}
        
        try:
            # Get user first
            user_response = self.db.client.table('users').select('id').eq('username', username).execute()
            
            if not user_response.data:
                logger.warning(f"User not found: {username}")
                return {'tier': 'free', 'status': 'active', 'ai_questions_used': 0}
            
            user_id = user_response.data[0]['id']
            
            # Get subscription
            sub_response = self.db.client.table('subscriptions').select('*').eq('user_id', user_id).execute()
            
            if sub_response.data:
                sub = sub_response.data[0]
                return {
                    'user_id': sub['user_id'],
                    'tier': sub['tier'],
                    'status': sub['status'],
                    'ai_questions_used': sub['ai_questions_used'],
                    'stripe_customer_id': sub.get('stripe_customer_id'),
                    'stripe_subscription_id': sub.get('stripe_subscription_id'),
                    'started_at': sub.get('started_at'),
                    'expires_at': sub.get('expires_at')
                }
            
            return {'tier': 'free', 'status': 'active', 'ai_questions_used': 0}
            
        except Exception as e:
            logger.error(f"Get subscription error: {e}")
            return {'tier': 'free', 'status': 'active', 'ai_questions_used': 0}
    
    def increment_ai_questions(self, username: str):
        """Increment AI questions count"""
        if not self.db.is_connected():
            logger.warning("Database not connected, cannot increment AI questions")
            return
        
        try:
            # Get user ID
            user_response = self.db.client.table('users').select('id').eq('username', username).execute()
            
            if not user_response.data:
                logger.warning(f"User not found: {username}")
                return
            
            user_id = user_response.data[0]['id']
            
            # Get current count
            sub_response = self.db.client.table('subscriptions').select('ai_questions_used').eq('user_id', user_id).execute()
            
            if not sub_response.data:
                logger.warning(f"Subscription not found for user: {username}")
                return
            
            current_count = sub_response.data[0]['ai_questions_used']
            
            # Increment
            self.db.client.table('subscriptions').update({
                'ai_questions_used': current_count + 1
            }).eq('user_id', user_id).execute()
            
            logger.info(f"Incremented AI questions for {username}: {current_count} -> {current_count + 1}")
            
        except Exception as e:
            logger.error(f"Increment AI questions error: {e}")
    
    def get_ai_questions_remaining(self, username: str) -> int:
        """Get remaining AI questions"""
        subscription = self.get_user_subscription(username)
        tier = subscription.get('tier', 'free')
        limit = self.TIERS[tier]['limits']['ai_questions']
        used = subscription.get('ai_questions_used', 0)
        return max(0, limit - used)
    
    def can_access_feature(self, username: str, feature: str) -> bool:
        """Check if user can access feature"""
        subscription = self.get_user_subscription(username)
        tier = subscription.get('tier', 'free')
        return self.TIERS[tier]['limits'].get(feature, False)
    
    def upgrade_to_pro(self, username: str, stripe_customer_id: str, stripe_subscription_id: str):
        """Upgrade user to Pro tier"""
        if not self.db.is_connected():
            logger.error("Database not connected, cannot upgrade user")
            return
        
        try:
            # Get user ID
            user_response = self.db.client.table('users').select('id').eq('username', username).execute()
            
            if not user_response.data:
                logger.error(f"User not found: {username}")
                return
            
            user_id = user_response.data[0]['id']
            
            # Update subscription
            self.db.client.table('subscriptions').update({
                'tier': 'pro',
                'status': 'active',
                'stripe_customer_id': stripe_customer_id,
                'stripe_subscription_id': stripe_subscription_id,
                'started_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=30)).isoformat(),
                'ai_questions_used': 0
            }).eq('user_id', user_id).execute()
            
            logger.info(f"Upgraded user to Pro: {username}")
            
        except Exception as e:
            logger.error(f"Upgrade to pro error: {e}")