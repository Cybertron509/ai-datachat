"""
Stripe payment handler
"""
import os
import streamlit as st
import logging

logger = logging.getLogger(__name__)


class StripeHandler:
    """Handle Stripe payment integration"""
    
    def __init__(self):
        """Initialize Stripe handler"""
        # Get from environment variables only (Render-compatible)
        self.publishable_key = os.getenv('STRIPE_PUBLISHABLE_KEY', '')
        self.secret_key = os.getenv('STRIPE_SECRET_KEY', '')
        self.webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', '')
        
        # Only initialize Stripe if keys exist
        if self.is_configured():
            try:
                import stripe
                stripe.api_key = self.secret_key
                self.stripe = stripe
                logger.info("Stripe initialized successfully")
            except ImportError:
                logger.warning("Stripe library not installed")
                self.stripe = None
        else:
            self.stripe = None
            logger.info("Stripe not configured - payment features disabled")
    
    def is_configured(self) -> bool:
        """Check if Stripe is properly configured"""
        return bool(self.publishable_key and self.secret_key)
    
    def get_publishable_key(self) -> str:
        """Get Stripe publishable key"""
        return self.publishable_key