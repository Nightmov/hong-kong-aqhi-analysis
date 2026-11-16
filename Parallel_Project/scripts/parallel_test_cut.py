import pandas as pd
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. Define a single subtask
# ==========================================
def process_single_file(file_path):
    try:
        if 'stations' in file_path.lower() or 'station.csv' in file_path.lower():
            return None 

        df = pd.read_csv(file_path)
        
        grouped = df.groupby(['station', 'yyyymm'])['aqhi']
        
        result = pd.DataFrame({
            'mean_aqhi': grouped.mean(),
            'max_aqhi': grouped.max(),
            'min_aqhi': grouped.min(),
            'std_aqhi': grouped.std(),
            'skew_aqhi': grouped.skew(),
            'estimated_count': df[df['is_estimated'] == 1].groupby(['station', 'yyyymm'])['aqhi'].count()
        }).reset_index()
        
        result['estimated_count'] = result['estimated_count'].fillna(0)
        return result
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# ==========================================
# 2. Main Program
# ==========================================
if __name__ == '__main__':
    data_dir = '../data/split_by_station' 
    stations_file = '../data/stations.csv' 
    
    # 1. Read the static station table
    try:
        df_stations = pd.read_csv(stations_file)
        df_stations = df_stations.rename(columns={'station_en': 'station'})
        
        # 💡 Add filtering: Remove invisible spaces before and after the string
        df_stations['station'] = df_stations['station'].astype(str).str.strip()

        # Force fix the issue of inconsistent names
        name_mapping = {
            'Central & Western': 'Central/Western',
            'Central and Western': 'Central/Western',
            'Tseung Kwan O': 'Tseung Kwan O', 
        }
        df_stations['station'] = df_stations['station'].replace(name_mapping)
    except Exception as e:
        print(f"Failed to read the station table: {e}")
        df_stations = None

    # Retrieve the list of original files (used to generate the final results)
    raw_files = glob.glob(os.path.join(data_dir, '*.csv'))
    raw_files = [f for f in raw_files if 'stations' not in f.lower()]

    if not raw_files:
        print(f"❌ No valid AQHI data CSV file found in {data_dir}!")
        exit()

    # 💡 File list specifically for performance testing
    test_files = raw_files
    print(f"✅ Successfully loaded files: There are now {len(test_files)} real files split by station.\n")

    # ==========================================
    # 3. Performance Testing Phase (Compare parallel performance on real data)
    # ==========================================
    test_cores = [c for c in [1, 2, 4, 8, 10] if c <= os.cpu_count()]
    all_results = []

    # Define the datasets to be compared
    datasets_to_test = [
        {"name": "1x_Raw (Real Small Data)", "files": raw_files}
    ]

    for dataset in datasets_to_test:
        data_name = dataset["name"]
        files_to_run = dataset["files"]
        
        print(f"\n================ Starting test dataset: {data_name} (Number of files: {len(files_to_run)}) ================")
        
        dataset_results = []
        for cores in test_cores:
            print(f"--- Testing with {cores} processes ---")
            start_time = time.time()
            
            if cores == 1:
                # Serial execution
                for f in files_to_run:
                    process_single_file(f)
            else:
                # Parallel execution
                with ProcessPoolExecutor(max_workers=cores) as executor:
                    for _ in executor.map(process_single_file, files_to_run):
                        pass
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"✅ Time taken with {cores} processes: {duration:.4f} seconds")
            
            dataset_results.append({'Cores': cores, 'Time_Seconds': duration})
        
        # Calculate the speedup for the current dataset
        baseline_time = dataset_results[0]['Time_Seconds']
        for res in dataset_results:
            res['Speedup'] = baseline_time / res['Time_Seconds']
            res['Dataset'] = data_name
            all_results.append(res)

    # ==========================================
    # Print and save the complete performance comparison report
    # ==========================================
    summary_df = pd.DataFrame(all_results)
    
    # Adjust column order to make the report more readable
    summary_df = summary_df[['Dataset', 'Cores', 'Time_Seconds', 'Speedup']]
    
    print("\n=========================================================")
    print("🏆 Final Performance Comparison Report (Amdahl's Law & Overhead)")
    print("=========================================================")
    print(summary_df.to_string(index=False))
    
    os.makedirs('../results', exist_ok=True)
    summary_df.to_csv('../results/speedup_comparison_results.csv', index=False)
    print("\n💾 Test data saved to ../results/speedup_comparison_results.csv\n")

    # ==========================================
    # 4. Final Data Generation Phase (Run only once on raw data to ensure clean results)
    # ==========================================
    print("--- Generating final merged results ---")
    
    # Process raw files using a 4-core process pool
    final_results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        map_results = list(executor.map(process_single_file, raw_files))
        final_results = [r for r in map_results if r is not None]

    if final_results and df_stations is not None:
        df_final = pd.concat(final_results, ignore_index=True)
        
        # Merge with station info
        df_merged = pd.merge(df_final, df_stations[['station', 'station_type', 'latitude', 'longitude', 'first_year']], 
                             on='station', how='left')
        
        # Check for unmatched stations
        unmatched = set(df_final['station'].unique()) - set(df_stations['station'].unique())
        if unmatched:
            print(f"⚠️ Warning: The following stations were not matched: {unmatched}")
        else:
            print("✅ All stations matched successfully!")

        # Group and summarize by station type
        type_stats = df_merged.groupby('station_type')[['mean_aqhi', 'max_aqhi']].mean().reset_index()
        
        print("\n=== Differences in Mean AQHI by Station Type ===")
        print(type_stats.to_string(index=False))
        
        # Save for other team members to use
        df_merged.to_csv('../results/merged_aqhi_station_data.csv', index=False)
        print("\n💾 Station info merged and saved to results/merged_aqhi_station_data.csv")
    else:
        print("⚠️ Merge skipped: Failed to generate valid results or station table not found.")