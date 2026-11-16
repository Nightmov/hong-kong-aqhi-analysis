import pandas as pd
import os

# 1. Path to your large file
input_file = '../data/aqhi_long.csv' 

# 2. New folder to store the partitioned files
output_dir = '../data/split_by_station'
os.makedirs(output_dir, exist_ok=True)

print("Reading the 177MB large file, please wait...")
df = pd.read_csv(input_file)

print(f"Read complete! Total rows: {len(df)}. Starting partitioning by station...")

# 3. Group by station
for station_name, group_df in df.groupby('station'):
    # Filter out possible null values
    if pd.isna(station_name):
        continue
        
    # Handle file names: Some station names contain slashes (e.g., Central/Western) and cannot be used directly as file names.
    # Replace slashes, spaces, and other special characters with underscores
    safe_name = str(station_name).replace('/', '_').replace(' ', '_')
    
    file_name = f"{safe_name}.csv"
    file_path = os.path.join(output_dir, file_name)
    
    # Save as an independent small CSV
    group_df.to_csv(file_path, index=False)
    print(f"Generated: {file_name} (Contains {len(group_df)} records)")

print(f"\n✅ Partitioning complete! All files have been saved to the {output_dir} directory.")