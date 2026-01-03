#!/usr/bin/env python3
"""
================================================================================
CRYSTAL VAULT - Stripe Payment Integration
================================================================================

Handles subscription payments for Crystal Vault SaaS.

Plans:
- Free: $0/month - 10 entries, 1 device
- Pro: $4.99/month - 1000 entries, 5 devices
- Enterprise: $9.99/month - Unlimited everything

Discoverer: Sir Charles Spikes
Date: December 24, 2025
================================================================================
"""

import os
from typing import Dict, Optional
from datetime import datetime

# Stripe configuration
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "sk_test_placeholder")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "whsec_placeholder")

# Price IDs (create these in Stripe Dashboard)
STRIPE_PRICES = {
    "pro_monthly": os.environ.get("STRIPE_PRICE_PRO", "price_pro_monthly"),
    "pro_yearly": os.environ.get("STRIPE_PRICE_PRO_YEARLY", "price_pro_yearly"),
    "enterprise_monthly": os.environ.get("STRIPE_PRICE_ENTERPRISE", "price_enterprise_monthly"),
    "enterprise_yearly": os.environ.get("STRIPE_PRICE_ENTERPRISE_YEARLY", "price_enterprise_yearly"),
}

# Simulated payment system (replace with real Stripe in production)
class PaymentManager:
    """
    Manages subscription payments.
    
    In production, replace with actual Stripe API calls:
    
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    """
    
    def __init__(self):
        self.subscriptions: Dict[str, dict] = {}  # user_email -> subscription
    
    def create_checkout_session(self, user_email: str, price_id: str, 
                                 success_url: str, cancel_url: str) -> Dict:
        """
        Create a Stripe Checkout session.
        
        In production:
        ```python
        session = stripe.checkout.Session.create(
            customer_email=user_email,
            payment_method_types=['card'],
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='subscription',
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return {"checkout_url": session.url, "session_id": session.id}
        ```
        """
        # Simulated response
        session_id = f"cs_{datetime.utcnow().timestamp()}"
        return {
            "checkout_url": f"/checkout/simulate?session={session_id}",
            "session_id": session_id,
            "message": "Stripe integration ready - configure STRIPE_SECRET_KEY for production"
        }
    
    def create_customer_portal(self, customer_id: str, return_url: str) -> Dict:
        """
        Create a Stripe Customer Portal session for managing subscriptions.
        
        In production:
        ```python
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return {"portal_url": session.url}
        ```
        """
        return {
            "portal_url": f"/portal/simulate?customer={customer_id}",
            "message": "Customer portal ready - configure Stripe for production"
        }
    
    def handle_webhook(self, payload: bytes, signature: str) -> Dict:
        """
        Handle Stripe webhook events.
        
        In production:
        ```python
        event = stripe.Webhook.construct_event(
            payload, signature, STRIPE_WEBHOOK_SECRET
        )
        
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            # Upgrade user to paid tier
            
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            # Downgrade user to free tier
        ```
        """
        return {"received": True}
    
    def get_subscription(self, user_email: str) -> Optional[Dict]:
        """Get user's current subscription."""
        return self.subscriptions.get(user_email)
    
    def upgrade_user(self, user_email: str, tier: str) -> Dict:
        """Upgrade user to a paid tier (simulated)."""
        self.subscriptions[user_email] = {
            "tier": tier,
            "status": "active",
            "started_at": datetime.utcnow().isoformat(),
            "current_period_end": "2026-02-03T00:00:00Z"  # 1 month from now
        }
        return {"success": True, "tier": tier}
    
    def cancel_subscription(self, user_email: str) -> Dict:
        """Cancel user's subscription."""
        if user_email in self.subscriptions:
            self.subscriptions[user_email]["status"] = "canceled"
            return {"success": True, "message": "Subscription will end at period end"}
        return {"success": False, "message": "No active subscription"}


# Singleton instance
payment_manager = PaymentManager()


# ================================================================================
# FastAPI Routes (add to main app.py)
# ================================================================================

"""
Add these routes to backend/app.py:

from backend.stripe_payments import payment_manager, STRIPE_PRICES

@app.post("/api/payments/create-checkout")
async def create_checkout(tier: str, user: dict = Depends(verify_token)):
    price_id = STRIPE_PRICES.get(f"{tier}_monthly")
    if not price_id:
        raise HTTPException(status_code=400, detail="Invalid tier")
    
    return payment_manager.create_checkout_session(
        user_email=user["email"],
        price_id=price_id,
        success_url=f"{request.base_url}app?payment=success",
        cancel_url=f"{request.base_url}pricing?payment=canceled"
    )

@app.post("/api/payments/portal")
async def customer_portal(user: dict = Depends(verify_token)):
    return payment_manager.create_customer_portal(
        customer_id=user["email"],
        return_url=f"{request.base_url}app"
    )

@app.post("/api/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    return payment_manager.handle_webhook(payload, signature)
"""


# ================================================================================
# Stripe Setup Instructions
# ================================================================================

SETUP_INSTRUCTIONS = """
================================================================================
STRIPE SETUP INSTRUCTIONS
================================================================================

1. CREATE STRIPE ACCOUNT
   - Go to https://stripe.com
   - Sign up for a free account
   - Get your API keys from Dashboard > Developers > API keys

2. CREATE PRODUCTS & PRICES
   In Stripe Dashboard > Products:
   
   Product: Crystal Vault Pro
   - Price: $4.99/month (save the price ID)
   - Price: $49.99/year (save the price ID)
   
   Product: Crystal Vault Enterprise  
   - Price: $9.99/month (save the price ID)
   - Price: $99.99/year (save the price ID)

3. SET ENVIRONMENT VARIABLES
   export STRIPE_SECRET_KEY=sk_live_xxxx
   export STRIPE_WEBHOOK_SECRET=whsec_xxxx
   export STRIPE_PRICE_PRO=price_xxxx
   export STRIPE_PRICE_PRO_YEARLY=price_xxxx
   export STRIPE_PRICE_ENTERPRISE=price_xxxx
   export STRIPE_PRICE_ENTERPRISE_YEARLY=price_xxxx

4. CONFIGURE WEBHOOK
   In Stripe Dashboard > Developers > Webhooks:
   - Add endpoint: https://yourdomain.com/api/webhooks/stripe
   - Select events:
     - checkout.session.completed
     - customer.subscription.updated
     - customer.subscription.deleted
     - invoice.payment_succeeded
     - invoice.payment_failed

5. TEST MODE
   - Use test API keys (sk_test_xxxx) for development
   - Test card: 4242 4242 4242 4242
   - Any future date, any CVC

6. GO LIVE
   - Complete Stripe account verification
   - Switch to live API keys
   - Update webhook endpoint

================================================================================
"""

if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)

