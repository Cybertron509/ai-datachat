"""
Subscription and Payment Management
Handles Stripe integration and feature gates
"""
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Optional
import json
from pathlib import Path


class SubscriptionManager:
    """Manage user subscriptions and feature access"""
    
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
        self.subscriptions_file = Path('subscriptions.json')
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create subscriptions file if it doesn't exist"""
        if not self.subscriptions_file.exists():
            with open(self.subscriptions_file, 'w') as f:
                json.dump({}, f)
    
    def get_user_subscription(self, username: str) -> Dict:
        """Get user's subscription details"""
        with open(self.subscriptions_file, 'r') as f:
            subscriptions = json.load(f)
        
        user_sub = subscriptions.get(username, {
            'tier': 'free',
            'status': 'active',
            'started_at': datetime.now().isoformat(),
            'expires_at': None,
            'stripe_customer_id': None,
            'stripe_subscription_id': None,
            'ai_questions_used': 0
        })
        
        if 'ai_questions_used' not in user_sub:
            user_sub['ai_questions_used'] = 0
        
        return user_sub
    
    def update_subscription(self, username: str, subscription_data: Dict):
        """Update user subscription"""
        with open(self.subscriptions_file, 'r') as f:
            subscriptions = json.load(f)
        
        subscriptions[username] = subscription_data
        
        with open(self.subscriptions_file, 'w') as f:
            json.dump(subscriptions, f, indent=2)
    
    def increment_ai_questions(self, username: str):
        """Increment AI questions used for user"""
        subscription = self.get_user_subscription(username)
        subscription['ai_questions_used'] = subscription.get('ai_questions_used', 0) + 1
        self.update_subscription(username, subscription)
    
    def can_access_feature(self, username: str, feature: str) -> bool:
        """Check if user can access a feature"""
        subscription = self.get_user_subscription(username)
        tier = subscription.get('tier', 'free')
        
        return self.TIERS[tier]['limits'].get(feature, False)
    
    def get_remaining_ai_questions(self, username: str) -> int:
        """Get remaining AI questions for user"""
        subscription = self.get_user_subscription(username)
        tier = subscription.get('tier', 'free')
        limit = self.TIERS[tier]['limits']['ai_questions']
        used = subscription.get('ai_questions_used', 0)
        
        return max(0, limit - used)
    
    def has_ai_questions_remaining(self, username: str) -> bool:
        """Check if user has AI questions remaining"""
        return self.get_remaining_ai_questions(username) > 0
    
    def upgrade_to_pro(self, username: str, stripe_customer_id: str, stripe_subscription_id: str):
        """Upgrade user to pro tier"""
        subscription = self.get_user_subscription(username)
        subscription['tier'] = 'pro'
        subscription['status'] = 'active'
        subscription['started_at'] = datetime.now().isoformat()
        subscription['expires_at'] = (datetime.now() + timedelta(days=30)).isoformat()
        subscription['stripe_customer_id'] = stripe_customer_id
        subscription['stripe_subscription_id'] = stripe_subscription_id
        subscription['ai_questions_used'] = 0
        
        self.update_subscription(username, subscription)