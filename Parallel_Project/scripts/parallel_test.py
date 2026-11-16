import pandas as pd
import glob
import os
import time
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# 1. Core computation logic (executed by sub-processes)
# ==========================================
def process_chunk(df):
    """
    Receives a DataFrame chunk and executes the original groupby logic.
    """
    try:
        if df.empty:
            return None
            
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
        print(f"❌ Error calculating data chunk: {e}")
        return None


# ==========================================
# 2. Main Program
# ==========================================
if __name__ == '__main__':
    data_dir = '../data/' 
    stations_file = os.path.join(data_dir, 'stations.csv') 
    # Replace with your actual 180MB single large file path
    big_file_path = os.path.join(data_dir, 'aqhi_long.csv') 
    
    # Define logical chunk size (approx. 100k rows per chunk, tune based on machine memory)
    CHUNK_SIZE = 100_000  

    # 2.1 Read static station table (kept unchanged)
    try:
        df_stations = pd.read_csv(stations_file)
        df_stations = df_stations.rename(columns={'station_en': 'station'})
        df_stations['station'] = df_stations['station'].astype(str).str.strip()
        name_mapping = {
            'Central & Western': 'Central/Western',
            'Central and Western': 'Central/Western',
            'Tseung Kwan O': 'Tseung Kwan O', 
        }
        df_stations['station'] = df_stations['station'].replace(name_mapping)
    except Exception as e:
        print(f"Failed to read station table: {e}")
        df_stations = None

    if not os.path.exists(big_file_path):
        print(f"❌ Large file not found: {big_file_path}, please modify the path.")
        exit()

    # ==========================================
    # 3. Performance Testing Phase
    # ==========================================
    print(f"\n================ Starting test for large file logical partitioning ================")
    print(f"File size: {os.path.getsize(big_file_path) / 1024 / 1024:.2f} MB")

    # Fix: Initialize the all_results list before testing begins
    all_results = []

    # 3.1 Baseline test: single-process direct read
    print("\n--- [1] Baseline Test: Pandas single-process direct read ---")
    gc.collect()
    start = time.time()
    
    df_full = pd.read_csv(big_file_path)
    res_serial = process_chunk(df_full)
    del df_full  # Free memory
    
    time_serial = time.time() - start
    print(f"✅ Single-process time taken: {time_serial:.4f} seconds")

    # Fix: Append single-process baseline result to the summary list
    all_results.append({
        'Dataset': 'Logical Sharding (Single File)', 
        'Cores': 1, 
        'Time_Seconds': time_serial, 
        'Speedup': 1.0
    })

    # 3.2 Logical partitioning + multi-process test
    test_cores = [c for c in [2, 4, 8, 10] if c <= os.cpu_count()]
    
    for cores in test_cores:
        print(f"\n--- [2] Logical partitioning + {cores} parallel processes ---")
        gc.collect()
        start = time.time()
        
        # Use chunksize to get a generator (data is not fully read yet, only the pointer)
        chunks = pd.read_csv(big_file_path, chunksize=CHUNK_SIZE)
        
        chunk_results = []
        # Submit data chunks to the process pool
        with ProcessPoolExecutor(max_workers=cores) as executor:
            # Submit all tasks
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            
            # Collect results
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    chunk_results.append(res)
        
        # Concatenate the results of all chunks
        df_concat = pd.concat(chunk_results, ignore_index=True)
        
        # Here we only demonstrate performance, so we directly concatenate
        res_parallel = df_concat
        
        time_parallel = time.time() - start
        speedup = time_serial / time_parallel
        
        print(f"✅ {cores} processes time taken: {time_parallel:.4f} seconds")
        print(f"🚀 Speedup: {speedup:.2f}x")

        # Fix: Append each multi-process test result to the list
        all_results.append({
            'Dataset': 'Logical Sharding (Single File)', 
            'Cores': cores, 
            'Time_Seconds': time_parallel, 
            'Speedup': speedup
        })

    # Fix: Move data summary and saving logic outside the for loop to avoid being overwritten each run
    summary_df = pd.DataFrame(all_results)
    # Adjust column order to make the report more readable
    summary_df = summary_df[['Dataset', 'Cores', 'Time_Seconds', 'Speedup']]
    
    os.makedirs('../results', exist_ok=True)
    # Fix: Change file name to logical
    save_path = '../results/speedup_comparison_results_logical.csv'
    summary_df.to_csv(save_path, index=False)
    print(f"\n💾 Test data saved to {save_path}\n")
        

    # ==========================================
    # 4. Final Data Generation (using multi-process streaming approach)
    # ==========================================
    print("\n--- Generating final merged results ---")
    
    chunks = pd.read_csv(big_file_path, chunksize=CHUNK_SIZE)
    final_results = []
    # Fix to 4 cores to run the final results
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                final_results.append(r)
                
    if final_results and df_stations is not None:
        # Concatenate chunk results
        df_final = pd.concat(final_results, ignore_index=True)
        
        # Merge station table
        df_merged = pd.merge(
            df_final, 
            df_stations[['station', 'station_type', 'latitude', 'longitude', 'first_year']], 
            on='station', how='left'
        )
        
        unmatched = set(df_final['station'].unique()) - set(df_stations['station'].unique())
        if unmatched:
            print(f"⚠️ Warning: The following stations were not matched: {unmatched}")
        else:
            print("✅ All stations matched successfully!")

        # Group and summarize by station type
        type_stats = df_merged.groupby('station_type')[['mean_aqhi', 'max_aqhi']].mean().reset_index()
        print("\n=== Differences in Mean AQHI by Station Type ===")
        print(type_stats.to_string(index=False))
        
        os.makedirs('../results', exist_ok=True)
        df_merged.to_csv('../results/merged_aqhi_station_data_logical.csv', index=False)
        print("\n💾 Saved to results/merged_aqhi_station_data_logical.csv")
    else:
        print("⚠️ Merge skipped: Failed to generate valid results or station table is missing.")