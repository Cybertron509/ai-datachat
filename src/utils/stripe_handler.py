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
    
    def create_checkout_session(self, username: str, email: str, price_id: str) -> str:
        """Create Stripe checkout session"""
        if not self.is_configured() or not self.stripe:
            logger.warning("Stripe not configured")
            return None
        
        try:
            session = self.stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url='https://ai-datachat.onrender.com/?success=true',
                cancel_url='https://ai-datachat.onrender.com/?canceled=true',
                client_reference_id=username,
                customer_email=email,
            )
            return session.url
        except Exception as e:
            logger.error(f"Stripe checkout error: {str(e)}")
            return None
    
    def get_checkout_button_html(self, checkout_url: str) -> str:
        """Get HTML for checkout button"""
        return f'''
        <a href="{checkout_url}" target="_blank" style="
            display: inline-block;
            background-color: #635BFF;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
            text-align: center;
        ">
            Subscribe Now with Stripe
        </a>
        '''