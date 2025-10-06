"""
Subscription Manager with Supabase Integration
Manages user tiers, feature access, and usage tracking
"""
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
from src.utils.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


class SubscriptionManager:
    """Manage user subscriptions and feature access"""
    
    TIERS = {
        'free': {
            'name': 'Free',
            'price': 0,
            'limits': {
                'ai_questions': 2,
                'forecasting': False,
                'scenarios': False,
                'trust_score': False,
                'reports': False
            },
            'features': [
                '✓ Upload CSV, Excel, JSON files',
                '✓ 6 interactive visualizations',
                '✓ Statistical analysis',
                '✓ Data filtering & cleaning',
                '✓ 2 AI chat questions (lifetime)',
                '✓ Export to CSV/Excel'
            ]
        },
        'pro': {
            'name': 'Pro',
            'price': 24.99,
            'limits': {
                'ai_questions': 999999,  # Effectively unlimited
                'forecasting': True,
                'scenarios': True,
                'trust_score': True,
                'reports': True
            },
            'features': [
                '✓ Everything in Free',
                '✓ Unlimited AI chat',
                '✓ Time-series forecasting (6-month predictions)',
                '✓ Scenario simulation (what-if analysis)',
                '✓ Data quality trust score',
                '✓ Narrative report generation',
                '✓ Priority support',
                '✓ 2GB file uploads'
            ]
        }
    }
    
    def __init__(self):
        try:
            self.supabase = SupabaseClient.get_client()
            self.use_database = True
        except Exception as e:
            logger.warning(f"Supabase not available, falling back to memory: {str(e)}")
            self.use_database = False
            self.memory_subscriptions = {}
    
    def get_user_subscription(self, username: str) -> Dict[str, Any]:
        """
        Get user subscription data
        Returns subscription dict with tier, usage, etc.
        """
        if self.use_database:
            try:
                result = self.supabase.table('subscriptions').select('*').eq('username', username).execute()
                
                if result.data:
                    return result.data[0]
                else:
                    # Create default subscription if not exists
                    default_sub = {
                        'username': username,
                        'tier': 'free',
                        'status': 'active',
                        'ai_questions_used': 0,
                        'started_at': datetime.now().isoformat(),
                        'expires_at': None,
                        'stripe_customer_id': None,
                        'stripe_subscription_id': None
                    }
                    
                    self.supabase.table('subscriptions').insert(default_sub).execute()
                    return default_sub
                    
            except Exception as e:
                logger.error(f"Error getting subscription from database: {str(e)}")
                return self._get_default_subscription(username)
        else:
            # Memory fallback
            if username not in self.memory_subscriptions:
                self.memory_subscriptions[username] = self._get_default_subscription(username)
            return self.memory_subscriptions[username]
    
    def _get_default_subscription(self, username: str) -> Dict[str, Any]:
        """Get default subscription structure"""
        return {
            'username': username,
            'tier': 'free',
            'status': 'active',
            'ai_questions_used': 0,
            'started_at': datetime.now().isoformat(),
            'expires_at': None,
            'stripe_customer_id': None,
            'stripe_subscription_id': None
        }
    
    def update_subscription(self, username: str, subscription_data: Dict[str, Any]) -> bool:
        """
        Update subscription data in database
        Returns True if successful
        """
        if self.use_database:
            try:
                subscription_data['updated_at'] = datetime.now().isoformat()
                
                self.supabase.table('subscriptions').update(
                    subscription_data
                ).eq('username', username).execute()
                
                return True
                
            except Exception as e:
                logger.error(f"Error updating subscription: {str(e)}")
                return False
        else:
            # Memory fallback
            self.memory_subscriptions[username] = subscription_data
            return True
    
    def get_remaining_ai_questions(self, username: str) -> int:
        """
        Get remaining AI questions for user
        Returns count of remaining questions
        """
        subscription = self.get_user_subscription(username)
        tier = subscription.get('tier', 'free')
        limit = self.TIERS[tier]['limits']['ai_questions']
        used = subscription.get('ai_questions_used', 0)
        
        return max(0, limit - used)
    
    def has_ai_questions_remaining(self, username: str) -> bool:
        """Check if user has AI questions remaining"""
        return self.get_remaining_ai_questions(username) > 0
    
    def increment_ai_questions(self, username: str) -> bool:
        """
        Increment AI question usage counter
        Returns True if successful
        """
        subscription = self.get_user_subscription(username)
        current_used = subscription.get('ai_questions_used', 0)
        subscription['ai_questions_used'] = current_used + 1
        
        return self.update_subscription(username, subscription)
    
    def can_access_feature(self, username: str, feature: str) -> bool:
        """
        Check if user can access a specific feature
        Features: 'forecasting', 'scenarios', 'trust_score', 'reports'
        Returns True if user has access
        """
        subscription = self.get_user_subscription(username)
        tier = subscription.get('tier', 'free')
        
        return self.TIERS[tier]['limits'].get(feature, False)
    
    def upgrade_to_pro(
        self, 
        username: str, 
        stripe_customer_id: str, 
        stripe_subscription_id: str
    ) -> bool:
        """
        Upgrade user to Pro tier
        Returns True if successful
        """
        subscription = self.get_user_subscription(username)
        subscription['tier'] = 'pro'
        subscription['status'] = 'active'
        subscription['started_at'] = datetime.now().isoformat()
        subscription['expires_at'] = (datetime.now() + timedelta(days=30)).isoformat()
        subscription['stripe_customer_id'] = stripe_customer_id
        subscription['stripe_subscription_id'] = stripe_subscription_id
        subscription['ai_questions_used'] = 0  # Reset on upgrade
        
        success = self.update_subscription(username, subscription)
        
        if success:
            logger.info(f"User '{username}' upgraded to Pro")
        
        return success
    
    def downgrade_to_free(self, username: str) -> bool:
        """
        Downgrade user to Free tier
        Returns True if successful
        """
        subscription = self.get_user_subscription(username)
        subscription['tier'] = 'free'
        subscription['status'] = 'active'
        subscription['expires_at'] = None
        subscription['stripe_customer_id'] = None
        subscription['stripe_subscription_id'] = None
        
        success = self.update_subscription(username, subscription)
        
        if success:
            logger.info(f"User '{username}' downgraded to Free")
        
        return success
    
    def cancel_subscription(self, username: str) -> bool:
        """
        Cancel user's subscription (keeps Pro until expiry)
        Returns True if successful
        """
        subscription = self.get_user_subscription(username)
        subscription['status'] = 'cancelled'
        
        success = self.update_subscription(username, subscription)
        
        if success:
            logger.info(f"Subscription cancelled for user '{username}'")
        
        return success
    
    def check_expired_subscriptions(self) -> int:
        """
        Check for expired Pro subscriptions and downgrade them
        Returns count of downgraded users
        """
        if not self.use_database:
            return 0
        
        try:
            # Get all Pro subscriptions with expiry dates in the past
            now = datetime.now().isoformat()
            
            result = self.supabase.table('subscriptions').select('username').eq(
                'tier', 'pro'
            ).lt('expires_at', now).execute()
            
            count = 0
            for user in result.data:
                if self.downgrade_to_free(user['username']):
                    count += 1
            
            if count > 0:
                logger.info(f"Downgraded {count} expired subscriptions")
            
            return count
            
        except Exception as e:
            logger.error(f"Error checking expired subscriptions: {str(e)}")
            return 0
    
    def get_subscription_stats(self) -> Dict[str, int]:
        """
        Get overall subscription statistics
        Returns dict with counts per tier
        """
        if not self.use_database:
            return {'free': 0, 'pro': 0}
        
        try:
            # Count free users
            free_result = self.supabase.table('subscriptions').select(
                'username', count='exact'
            ).eq('tier', 'free').execute()
            
            # Count pro users
            pro_result = self.supabase.table('subscriptions').select(
                'username', count='exact'
            ).eq('tier', 'pro').execute()
            
            return {
                'free': free_result.count if hasattr(free_result, 'count') else 0,
                'pro': pro_result.count if hasattr(pro_result, 'count') else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting subscription stats: {str(e)}")
            return {'free': 0, 'pro': 0}