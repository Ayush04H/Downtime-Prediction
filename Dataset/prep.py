import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2500 records
n_records = 2500

# Generate base data
data = {
    'Machine_ID': np.random.choice(['M' + str(i) for i in range(1, 11)], n_records),
    'Timestamp': [(datetime(2024, 1, 1) + timedelta(hours=x)) for x in range(n_records)],
    'Temperature': np.random.normal(75, 15, n_records),  # Normal distribution around 75°C
    'Vibration': np.random.normal(0.5, 0.2, n_records),  # Vibration levels
    'Pressure': np.random.normal(100, 20, n_records),    # Operating pressure
    'Run_Time': np.random.uniform(0, 24, n_records),     # Hours per day
    'Oil_Level': np.random.uniform(0.7, 1.0, n_records), # Oil level percentage
    'Power_Consumption': np.random.normal(80, 10, n_records), # Power consumption percentage
    'Product_Rate': np.random.normal(100, 15, n_records),  # Products per hour
    'Maintenance_Due': np.random.choice([0, 1], n_records, p=[0.9, 0.1])  # Binary flag for maintenance
}

# Create DataFrame
df = pd.DataFrame(data)

# Add some realistic constraints and relationships
df['Temperature'] = np.where(df['Run_Time'] > 20, 
                           df['Temperature'] * 1.2,  # Higher temps for longer runs
                           df['Temperature'])

# Create Downtime_Flag based on multiple conditions
conditions = (
    (df['Temperature'] > 95) |  # High temperature
    (df['Vibration'] > 0.8) |   # High vibration
    (df['Oil_Level'] < 0.75) |  # Low oil
    (df['Maintenance_Due'] == 1) # Maintenance needed
)
df['Downtime_Flag'] = conditions.astype(int)

# Add Downtime (Yes/No) column
df['Downtime'] = df['Downtime_Flag'].map({0: 'No', 1: 'Yes'})

# Add Quality_Score (inverse relationship with extreme conditions)
df['Quality_Score'] = 100 - (
    (df['Temperature'] - 75).abs() * 0.5 +
    (df['Vibration'] - 0.5).abs() * 20 +
    (df['Pressure'] - 100).abs() * 0.2
).clip(0, 100)

# Add Equipment_Age (days since installation)
df['Equipment_Age'] = df.groupby('Machine_ID').cumcount() + \
                     np.random.randint(100, 1000, len(df))

# Clean up and round values
df['Temperature'] = df['Temperature'].round(2)
df['Vibration'] = df['Vibration'].round(3)
df['Pressure'] = df['Pressure'].round(2)
df['Run_Time'] = df['Run_Time'].round(2)
df['Oil_Level'] = df['Oil_Level'].round(3)
df['Power_Consumption'] = df['Power_Consumption'].round(2)
df['Product_Rate'] = df['Product_Rate'].round(2)
df['Quality_Score'] = df['Quality_Score'].round(2)

# Sort by Timestamp
df = df.sort_values('Timestamp')

# Display first few rows and dataset info
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Optional: Save to CSV
df.to_csv('manufacturing_data.csv', index=False)

# Display downtime distribution
print("\nDowntime Distribution:")
print(df['Downtime'].value_counts())