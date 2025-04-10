# fetch_google_play_reviews.py
import pandas as pd
from google_play_scraper import reviews, Sort
import sys
import time
import os # Import os to check for file existence

# --- Configuration ---
app_id = 'chat.sekai.app'
country_code = 'us'
language_code = 'en'
# Fetch a smaller batch frequently rather than a huge batch infrequently
fetch_count = 500 # How many recent reviews to check on each run
output_file = 'sekai_google_play_reviews.csv'
# Unique ID column provided by the scraper
id_column = 'reviewId'
# Date column to sort by if needed (though scraper sorts by newest)
date_column = 'at'

print(f"Starting incremental fetch for {app_id}...")

# --- Load Existing Reviews ---
existing_df = pd.DataFrame()
existing_ids = set()
latest_timestamp = None

if os.path.exists(output_file):
    try:
        print(f"Loading existing reviews from {output_file}...")
        existing_df = pd.read_csv(output_file)
        if not existing_df.empty and id_column in existing_df.columns:
            existing_ids = set(existing_df[id_column].astype(str)) # Ensure IDs are strings for comparison
            # Convert 'at' column to datetime if it exists, handling potential errors
            if date_column in existing_df.columns:
                 existing_df[date_column] = pd.to_datetime(existing_df[date_column], errors='coerce')
                 latest_timestamp = existing_df[date_column].dropna().max() # Get the latest valid timestamp
            print(f"Loaded {len(existing_df)} existing reviews. Found {len(existing_ids)} unique IDs.")
            if latest_timestamp:
                print(f"Latest existing review timestamp: {latest_timestamp}")
        else:
            print(f"Existing file {output_file} is empty or missing '{id_column}'. Will fetch initial batch.")
            existing_df = pd.DataFrame() # Ensure it's an empty DataFrame
    except Exception as e:
        print(f"Error loading existing reviews from {output_file}: {e}. Will proceed as if fetching for the first time.")
        existing_df = pd.DataFrame() # Reset on error
        existing_ids = set()
        latest_timestamp = None
else:
    print(f"No existing review file found at {output_file}. Will fetch initial batch.")

# --- Fetch Latest Batch ---
print(f"Fetching the latest {fetch_count} reviews from Google Play...")
try:
    result, continuation_token = reviews(
        app_id,
        lang=language_code,
        country=country_code,
        sort=Sort.NEWEST,
        count=fetch_count,
        filter_score_with=None
    )

    if not result:
        print("No reviews fetched in this batch.")
        sys.exit() # Nothing to do

    fetched_df = pd.DataFrame(result)
    print(f"Fetched {len(fetched_df)} reviews in this batch.")

    # --- Data Cleaning (similar to before) ---
    if 'content' in fetched_df.columns:
        fetched_df['content'] = fetched_df['content'].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
        fetched_df.rename(columns={'content': 'review'}, inplace=True)

    # Ensure the ID column is string type for comparison
    if id_column in fetched_df.columns:
         fetched_df[id_column] = fetched_df[id_column].astype(str)
    else:
         print(f"Error: Fetched data is missing the ID column '{id_column}'. Cannot determine new reviews reliably.")
         sys.exit()

    # Convert date column for potential filtering (though ID check is primary)
    if date_column in fetched_df.columns:
        fetched_df[date_column] = pd.to_datetime(fetched_df[date_column], errors='coerce')


    # --- Identify New Reviews ---
    new_reviews_df = fetched_df[~fetched_df[id_column].isin(existing_ids)]
    new_count = len(new_reviews_df)

    if new_count > 0:
        print(f"Identified {new_count} new reviews.")

        # --- Append New Reviews ---
        # If the file exists, append without header; otherwise, write with header.
        write_header = not os.path.exists(output_file) or existing_df.empty
        save_mode = 'a' if not write_header else 'w'

        # Ensure columns match existing file if appending
        if save_mode == 'a' and not existing_df.empty:
             # Reorder columns of new_reviews_df to match existing_df
             # Only include columns that are present in the existing dataframe
             cols_to_save = [col for col in existing_df.columns if col in new_reviews_df.columns]
             new_reviews_to_save = new_reviews_df[cols_to_save]
        else:
             # If writing new file or existing was empty, save all columns from fetched data
             new_reviews_to_save = new_reviews_df

        print(f"Appending {new_count} new reviews to {output_file}...")
        new_reviews_to_save.to_csv(output_file, mode=save_mode, header=write_header, index=False, encoding='utf-8-sig')

        print(f"Successfully appended {new_count} new reviews.")
        print("\nPreview of newly added reviews:")
        preview_cols = ['reviewId', 'review', 'score', 'at', 'userName']
        print(new_reviews_to_save[[col for col in preview_cols if col in new_reviews_to_save.columns]].head().to_string())

    else:
        print("No new reviews found in this batch.")

except Exception as e:
    print(f"\nAn error occurred during fetching or processing: {e}")
    # Consider adding more specific error handling if needed

print(f"Finished incremental fetch for {app_id}.")

# Removed the previous direct save and preview logic
# ... existing code ...
# Save the DataFrame to a CSV file
# output_file = 'sekai_google_play_reviews.csv'
# df_reviews.to_csv(output_file, index=False, encoding='utf-8-sig')
#
# print(f"\nSuccessfully fetched {len(df_reviews)} reviews and saved them to {output_file}")
# Print the first 5 rows to give a preview
# print("\nFirst 5 reviews (Google Play):")
# Select relevant columns for preview to avoid excessive width
# preview_cols = ['review', 'score', 'at', 'userName']
# print(df_reviews[preview_cols].head().to_string())

# except Exception as e:
#    print(f"\nAn error occurred: {e}")
#    print("This could be due to network issues, changes in the Play Store structure, or the app not being found.") 