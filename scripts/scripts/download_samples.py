"""
Download sample datasets for testing
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Create scripts directory if running directly
script_dir = Path(__file__).parent
samples_dir = script_dir.parent / "data" / "samples"
samples_dir.mkdir(parents=True, exist_ok=True)

print(f"Creating sample datasets in: {samples_dir}")


def create_sales_data():
    """Create a sample sales dataset"""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    regions = ['North', 'South', 'East', 'West']
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    
    n_records = 1000
    
    df = pd.DataFrame({
        'date': np.random.choice(dates, n_records),
        'region': np.random.choice(regions, n_records),
        'product': np.random.choice(products, n_records),
        'quantity': np.random.randint(1, 100, n_records),
        'price': np.random.uniform(10, 500, n_records).round(2),
        'discount': np.random.uniform(0, 0.3, n_records).round(2)
    })
    
    # Calculate revenue
    df['revenue'] = (df['quantity'] * df['price'] * (1 - df['discount'])).round(2)
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 50), 'discount'] = np.nan
    
    file_path = samples_dir / "sales_data.csv"
    df.to_csv(file_path, index=False)
    print(f"✓ Created sales_data.csv ({len(df)} rows)")
    return file_path


def create_customer_data():
    """Create a sample customer dataset"""
    np.random.seed(42)
    
    n_customers = 500
    
    df = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'age': np.random.randint(18, 75, n_customers),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_customers),
        'income': np.random.randint(20000, 150000, n_customers),
        'purchase_count': np.random.randint(1, 50, n_customers),
        'total_spent': np.random.uniform(100, 10000, n_customers).round(2),
        'satisfaction': np.random.randint(1, 6, n_customers),
        'membership': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_customers)
    })
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 30), 'income'] = np.nan
    
    file_path = samples_dir / "customer_data.csv"
    df.to_csv(file_path, index=False)
    print(f"✓ Created customer_data.csv ({len(df)} rows)")
    return file_path


def create_employee_data():
    """Create a sample employee dataset"""
    np.random.seed(42)
    
    n_employees = 300
    departments = ['Sales', 'Marketing', 'Engineering', 'HR', 'Finance']
    
    df = pd.DataFrame({
        'employee_id': range(1001, 1001 + n_employees),
        'name': [f'Employee_{i}' for i in range(1, n_employees + 1)],
        'department': np.random.choice(departments, n_employees),
        'salary': np.random.randint(40000, 150000, n_employees),
        'years_experience': np.random.randint(0, 30, n_employees),
        'performance_rating': np.random.uniform(2.0, 5.0, n_employees).round(1),
        'remote': np.random.choice([True, False], n_employees)
    })
    
    file_path = samples_dir / "employee_data.xlsx"
    df.to_excel(file_path, index=False)
    print(f"✓ Created employee_data.xlsx ({len(df)} rows)")
    return file_path


def create_product_inventory():
    """Create a sample product inventory dataset"""
    np.random.seed(42)
    
    n_products = 200
    categories = ['Electronics', 'Clothing', 'Food', 'Home & Garden', 'Sports']
    
    df = pd.DataFrame({
        'product_id': range(1, n_products + 1),
        'product_name': [f'Product_{i}' for i in range(1, n_products + 1)],
        'category': np.random.choice(categories, n_products),
        'stock_quantity': np.random.randint(0, 500, n_products),
        'reorder_level': np.random.randint(10, 100, n_products),
        'unit_cost': np.random.uniform(5, 200, n_products).round(2),
        'unit_price': np.random.uniform(10, 400, n_products).round(2),
        'supplier': np.random.choice(['Supplier A', 'Supplier B', 'Supplier C'], n_products)
    })
    
    # Calculate profit margin
    df['profit_margin'] = ((df['unit_price'] - df['unit_cost']) / df['unit_price'] * 100).round(2)
    
    file_path = samples_dir / "product_inventory.csv"
    df.to_csv(file_path, index=False)
    print(f"✓ Created product_inventory.csv ({len(df)} rows)")
    return file_path


def create_weather_data():
    """Create a sample weather dataset"""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    
    records = []
    for city in cities:
        for date in dates:
            records.append({
                'date': date,
                'city': city,
                'temperature': np.random.uniform(-10, 40, 1)[0].round(1),
                'humidity': np.random.uniform(20, 100, 1)[0].round(1),
                'precipitation': np.random.uniform(0, 50, 1)[0].round(1),
                'wind_speed': np.random.uniform(0, 30, 1)[0].round(1)
            })
    
    df = pd.DataFrame(records)
    
    file_path = samples_dir / "weather_data.csv"
    df.to_csv(file_path, index=False)
    print(f"✓ Created weather_data.csv ({len(df)} rows)")
    return file_path


def create_json_sample():
    """Create a sample JSON dataset"""
    data = {
        "users": [
            {"id": i, "username": f"user{i}", "email": f"user{i}@example.com", 
             "age": np.random.randint(18, 65), "active": np.random.choice([True, False])}
            for i in range(1, 101)
        ]
    }
    
    file_path = samples_dir / "users_data.json"
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Created users_data.json (100 users)")
    return file_path


def main():
    """Create all sample datasets"""
    print("\n" + "="*50)
    print("Creating Sample Datasets for AI DataChat")
    print("="*50 + "\n")
    
    try:
        create_sales_data()
        create_customer_data()
        create_employee_data()
        create_product_inventory()
        create_weather_data()
        create_json_sample()
        
        print("\n" + "="*50)
        print(f"✓ All sample datasets created successfully!")
        print(f"Location: {samples_dir.absolute()}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error creating datasets: {str(e)}")
        raise


if __name__ == "__main__":
    main()