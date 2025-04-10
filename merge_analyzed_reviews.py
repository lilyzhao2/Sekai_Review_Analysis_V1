# merge_analyzed_reviews.py
import pandas as pd
import sys

# --- Configuration ---
IOS_ANALYZED_FILE = 'sekai_reviews_analyzed.csv'
ANDROID_ANALYZED_FILE = 'sekai_google_play_reviews_analyzed.csv'
OUTPUT_FILE = 'sekai_all_reviews_analyzed.csv'
# --- End Configuration ---

def load_and_prepare(filepath, platform_name):
    """Loads a CSV, adds platform column, handles potential errors, selects/renames columns."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} rows from {filepath}")
        df['platform'] = platform_name

        # --- Column Standardization ---
        # Rename date columns to a common name
        if platform_name == 'iOS' and 'date' in df.columns:
            df.rename(columns={'date': 'review_date'}, inplace=True)
        elif platform_name == 'Android' and 'at' in df.columns:
            df.rename(columns={'at': 'review_date'}, inplace=True)

        # Select common/important columns (adjust as needed)
        # We prioritize the analyzed fields, platform, and basic review info
        common_columns = [
            'review', 'review_date', 'userName', 'platform', 
            'topic', 'sentiment', 'pain_points', 'mentioned_ui_elements', 'summary'
        ]
        
        # Keep only columns that exist in the DataFrame
        columns_to_keep = [col for col in common_columns if col in df.columns]
        df_selected = df[columns_to_keep]
        
        # Convert date column to datetime objects, handling potential errors
        if 'review_date' in df_selected.columns:
            df_selected['review_date'] = pd.to_datetime(df_selected['review_date'], errors='coerce')
            
        print(f"Prepared {platform_name} data with columns: {df_selected.columns.tolist()}")
        return df_selected

    except FileNotFoundError:
        print(f"Error: File not found - {filepath}. Skipping this file.")
        return None
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    df_ios = load_and_prepare(IOS_ANALYZED_FILE, 'iOS')
    df_android = load_and_prepare(ANDROID_ANALYZED_FILE, 'Android')

    # Create a list of DataFrames that were loaded successfully
    all_dfs = [df for df in [df_ios, df_android] if df is not None]

    if not all_dfs:
        print("Error: No dataframes were loaded successfully. Exiting.")
        sys.exit(1)
    elif len(all_dfs) < 2:
        print("Warning: Only one platform's data was loaded. The output file will contain only that data.")

    # Concatenate the DataFrames
    # ignore_index=True creates a new continuous index for the combined DataFrame
    df_combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\nCombined data contains {len(df_combined)} rows.")

    # Save the combined data
    try:
        df_combined.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"Successfully merged data saved to {OUTPUT_FILE}")
        print("\nColumns in merged file:")
        print(df_combined.columns.tolist())
        print("\nSample rows (showing platform):")
        # Show a sample ensuring both platforms are likely visible if present
        print(df_combined[[ 'platform', 'review_date','userName', 'sentiment', 'review']].head().to_string())
        if len(df_combined) > 5:
             print("...")
             print(df_combined[[ 'platform', 'review_date','userName', 'sentiment', 'review']].tail().to_string())

    except Exception as e:
        print(f"Error saving combined data to {OUTPUT_FILE}: {e}") 