# fetch_reviews.py
import pandas as pd
from app_store_scraper import AppStore
import sys
import os
from datetime import datetime, timedelta

# --- Configuration ---
country_code = 'us'
app_name = 'sekai'
app_id = '6504506653'
# Fetch reviews potentially added in the last N days as a buffer, 
# though the `after` parameter is the primary mechanism.
# The `how_many` parameter is less relevant when using `after`, 
# but we set a reasonable limit.
fetch_limit = 1000 # Max reviews to check per run if `after` fails
sleep_timer = 5 # Seconds between requests
output_file = 'sekai_app_store_reviews.csv'
# The App Store scraper doesn't have a stable unique ID across fetches,
# so we rely primarily on the 'date' column.
date_column = 'date' # Date column used for incremental fetching

print(f"Starting incremental fetch for {app_name} (ID: {app_id})...")

# --- Load Existing Reviews ---
existing_df = pd.DataFrame()
latest_review_date = None
existing_review_hashes = set() # Use a hash of content/user/date to check for duplicates

if os.path.exists(output_file):
    try:
        print(f"Loading existing reviews from {output_file}...")
        existing_df = pd.read_csv(output_file)
        if not existing_df.empty:
            # Convert date column and find the latest
            if date_column in existing_df.columns:
                existing_df[date_column] = pd.to_datetime(existing_df[date_column], errors='coerce')
                latest_review_date = existing_df[date_column].dropna().max()
            
            # Create a simple hash for existing reviews to help identify duplicates
            # as App Store doesn't provide a stable ID.
            hash_cols = ['userName', 'rating', 'title', 'review', date_column]
            cols_present = [col for col in hash_cols if col in existing_df.columns]
            if cols_present:
                existing_review_hashes = set(existing_df[cols_present].astype(str).agg('_'.join, axis=1))
            
            print(f"Loaded {len(existing_df)} existing reviews.")
            if latest_review_date:
                print(f"Latest existing review date: {latest_review_date}")
            print(f"Generated {len(existing_review_hashes)} hashes for duplicate checking.")
        else:
            print(f"Existing file {output_file} is empty. Will fetch initial batch.")
            existing_df = pd.DataFrame()
    except Exception as e:
        print(f"Error loading existing reviews from {output_file}: {e}. Will proceed as if fetching for the first time.")
        existing_df = pd.DataFrame()
        latest_review_date = None
        existing_review_hashes = set()
else:
    print(f"No existing review file found at {output_file}. Will fetch initial batch.")

# --- Fetch Latest Reviews ---
# Fetch reviews after the latest one we have, adding a small buffer just in case.
fetch_after_date = None
if latest_review_date:
    # Fetch reviews slightly before the last known date to avoid missing any due to timing/rounding.
    fetch_after_date = latest_review_date - timedelta(days=1) 
    print(f"Fetching reviews after: {fetch_after_date}")
else:
    print(f"Fetching initial batch of up to {fetch_limit} reviews...")

try:
    store = AppStore(country=country_code, app_name=app_name, app_id=app_id)
    
    # Use the 'after' parameter if we have a date, otherwise fetch a limited number.
    store.review(
        after=fetch_after_date, 
        how_many=fetch_limit if fetch_after_date is None else None, # how_many is ignored if after is set
        sleep=sleep_timer
    ) 

    reviews_data = store.reviews

    if not reviews_data:
        print("No new reviews fetched in this batch.")
        sys.exit()

    fetched_df = pd.DataFrame(reviews_data)
    print(f"Fetched {len(fetched_df)} reviews in this batch.")

    # --- Data Cleaning ---
    if 'review' in fetched_df.columns:
        fetched_df['review'] = fetched_df['review'].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
    if date_column in fetched_df.columns:
        fetched_df[date_column] = pd.to_datetime(fetched_df[date_column], errors='coerce')
        # Drop rows where date conversion failed, as they can't be reliably compared
        fetched_df.dropna(subset=[date_column], inplace=True)
        
    # --- Identify New Reviews (using hash comparison) ---
    if not fetched_df.empty:
        # Create hash for newly fetched reviews
        hash_cols = ['userName', 'rating', 'title', 'review', date_column]
        cols_present = [col for col in hash_cols if col in fetched_df.columns]
        if cols_present:
            fetched_df['review_hash'] = fetched_df[cols_present].astype(str).agg('_'.join, axis=1)
            
            # Filter out reviews whose hash matches an existing hash
            new_reviews_df = fetched_df[~fetched_df['review_hash'].isin(existing_review_hashes)].copy()
            # Drop the temporary hash column
            new_reviews_df.drop(columns=['review_hash'], inplace=True, errors='ignore')
        else:
            print("Warning: Could not generate hashes for fetched reviews. Appending all fetched reviews.")
            new_reviews_df = fetched_df # Fallback if hashing fails
    else:
        new_reviews_df = pd.DataFrame() # No valid fetched reviews

    new_count = len(new_reviews_df)

    if new_count > 0:
        print(f"Identified {new_count} potentially new reviews after hash comparison.")

        # --- Append New Reviews ---
        write_header = not os.path.exists(output_file) or existing_df.empty
        save_mode = 'a' if not write_header else 'w'

        # Align columns before saving
        if save_mode == 'a' and not existing_df.empty:
            cols_to_save = [col for col in existing_df.columns if col in new_reviews_df.columns]
            new_reviews_to_save = new_reviews_df[cols_to_save]
        else:
            new_reviews_to_save = new_reviews_df

        print(f"Appending {new_count} new reviews to {output_file}...")
        new_reviews_to_save.to_csv(output_file, mode=save_mode, header=write_header, index=False, encoding='utf-8-sig')

        print(f"Successfully appended {new_count} new reviews.")
        print("\nPreview of newly added reviews:")
        preview_cols = ['date', 'userName', 'rating', 'title', 'review']
        print(new_reviews_to_save[[col for col in preview_cols if col in new_reviews_to_save.columns]].head().to_string())

    else:
        print("No new reviews found in this batch after checking against existing ones.")

except Exception as e:
    print(f"\nAn error occurred during fetching or processing: {e}")

print(f"Finished incremental fetch for {app_name} (App Store).")