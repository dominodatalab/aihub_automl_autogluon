import pandas as pd

# Load the dataset
file_path = 'data/obfuscated_6_series_2021-2024_billing_and_holidays.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Subset the data for the product corresponding to the series_id options are [Narnia_OptiGlimpse,Narnia_LensLogic,Atlantis_IrisInnovations,Atlantis_IrisInnovations,Atlantis_IrisInnovations]
product_name = "Narnia_OptiGlimpse"  # Change this to the product name you want to subset
subset_data = data[data['series_id'] == product_name]

# Display the first few rows of the subset data to verify
print(subset_data.head())

# Optionally, save the subset to a new CSV file
subset_file_path = f'data/{product_name}.csv'
subset_data.to_csv(subset_file_path, index=False)

print(f"Subset data saved to {subset_file_path}")