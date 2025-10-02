"""
Stripe Payment Integration
Handles payment processing and webhook events
"""
import streamlit as st
import os


class StripeHandler:
    """Handle Stripe payment processing"""
    
    def __init__(self):
        # In production, set these in Streamlit secrets
        self.publishable_key = os.getenv('STRIPE_PUBLISHABLE_KEY', '')
        self.secret_key = os.getenv('STRIPE_SECRET_KEY', '')
        self.webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', '')
        
        # Check if stripe is available
        try:
            import stripe
            self.stripe = stripe
            if self.secret_key:
                self.stripe.api_key = self.secret_key
        except ImportError:
            self.stripe = None
    
    def is_configured(self) -> bool:
        """Check if Stripe is properly configured"""
        return (self.stripe is not None and 
                bool(self.publishable_key) and 
                bool(self.secret_key))
    
    def create_checkout_session(self, username: str, email: str, price_id: str) -> str:
        """Create Stripe checkout session"""
        if not self.is_configured():
            return None
        
        try:
            session = self.stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=f'{st.secrets.get("APP_URL", "")}/success?session_id={{CHECKOUT_SESSION_ID}}',
                cancel_url=f'{st.secrets.get("APP_URL", "")}/cancel',
                customer_email=email,
                client_reference_id=username,
                metadata={'username': username}
            )
            
            return session.url
        except Exception as e:
            st.error(f"Error creating checkout session: {str(e)}")
            return None
    
    def get_checkout_button_html(self, session_url: str) -> str:
        """Generate Stripe checkout button HTML"""
        return f"""
        <a href="{session_url}" target="_blank">
            <button style="
                background-color: #635BFF;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 4px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
            ">
                Subscribe to Pro - $24.99/month
            </button>
        </a>
        """