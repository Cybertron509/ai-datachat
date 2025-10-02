import pandas as pd
import numpy as np

np.random.seed(42)

# Generate 50,000 records
n_records = 50000

print("Generating complex e-commerce dataset...")

# Customer demographics
data = {
    'customer_id': np.arange(1, n_records + 1),
    'age': np.random.normal(38, 12, n_records).clip(18, 75).astype(int),
    'gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_records, p=[0.48, 0.48, 0.04]),
    'country': np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia', 'Japan'], 
                               n_records, p=[0.35, 0.15, 0.12, 0.10, 0.10, 0.08, 0.10]),
    'account_age_days': np.random.exponential(365, n_records).clip(1, 3650).astype(int),
    'membership_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 
                                       n_records, p=[0.50, 0.30, 0.15, 0.05])
}

# Purchase behavior with realistic correlations
base_purchases = np.random.poisson(8, n_records)
tier_multiplier = np.where(data['membership_tier'] == 'Platinum', 2.5,
                          np.where(data['membership_tier'] == 'Gold', 1.8,
                          np.where(data['membership_tier'] == 'Silver', 1.2, 1.0)))
data['total_purchases'] = (base_purchases * tier_multiplier).astype(int)

# Revenue
base_revenue = data['total_purchases'] * np.random.uniform(25, 150, n_records)
tier_revenue_mult = np.where(data['membership_tier'] == 'Platinum', 1.5,
                             np.where(data['membership_tier'] == 'Gold', 1.3,
                             np.where(data['membership_tier'] == 'Silver', 1.1, 1.0)))
data['total_revenue'] = base_revenue * tier_revenue_mult
data['avg_order_value'] = data['total_revenue'] / np.maximum(data['total_purchases'], 1)

# Customer satisfaction and behavior
data['return_rate'] = np.random.beta(2, 20, n_records) * 100
data['satisfaction_score'] = (100 - data['return_rate'] * 0.5 + np.random.normal(0, 10, n_records)).clip(1, 100)
data['email_open_rate'] = np.random.beta(3, 5, n_records) * 100
data['click_through_rate'] = data['email_open_rate'] * np.random.uniform(0.15, 0.35, n_records)

# Website behavior
data['sessions_per_month'] = np.random.gamma(3, 2, n_records).clip(0, 50)
data['avg_session_duration_min'] = np.random.gamma(5, 2, n_records).clip(0.5, 30)
data['pages_per_session'] = np.random.gamma(4, 2, n_records).clip(1, 20)

# Product preferences
data['favorite_category'] = np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty'], 
                                            n_records, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
data['cart_abandonment_rate'] = (100 - data['satisfaction_score'] * 0.3 + np.random.normal(20, 10, n_records)).clip(5, 95)
data['wishlist_items'] = np.random.poisson(5, n_records)

# Customer service
data['support_tickets'] = np.random.poisson(2, n_records)
data['avg_resolution_time_hrs'] = np.random.gamma(2, 12, n_records).clip(1, 72)

# Engagement
data['promo_code_usage'] = np.random.binomial(data['total_purchases'], 0.3)
data['referrals_made'] = np.random.poisson(1.5, n_records)
data['device_type'] = np.random.choice(['Desktop', 'Mobile', 'Tablet'], n_records, p=[0.40, 0.50, 0.10])
data['social_media_connected'] = np.random.choice([0, 1], n_records, p=[0.35, 0.65])

# Advanced metrics
data['recency_days'] = np.random.exponential(30, n_records).clip(0, 365).astype(int)
data['churn_risk_score'] = (
    (data['recency_days'] / 365 * 30) +
    (data['cart_abandonment_rate'] * 0.2) +
    ((100 - data['satisfaction_score']) * 0.3) +
    (data['return_rate'] * 0.2) +
    np.random.normal(0, 10, n_records)
).clip(0, 100)

data['customer_lifetime_value'] = (
    data['total_revenue'] * 
    (1 + data['account_age_days'] / 1000) * 
    (1 - data['churn_risk_score'] / 200) +
    np.random.normal(0, 100, n_records)
).clip(0)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
filename = 'complex_ecommerce_dataset.csv'
df.to_csv(filename, index=False)

print(f"Dataset generated successfully!")
print(f"Filename: {filename}")
print(f"Records: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"File size: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
print("\nColumn names:")
for col in df.columns:
    print(f"  - {col}")