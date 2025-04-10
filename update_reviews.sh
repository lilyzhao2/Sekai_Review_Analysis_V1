#!/bin/zsh

# update_reviews.sh
# This script runs the review fetching and analysis process.

# Ensure we are in the script's directory
cd "$(dirname "$0")" || exit

echo "Starting review update process..."

# Step 1: Fetch new App Store reviews
echo "\n---> Running App Store fetch script (fetch_reviews.py)..."
python3 fetch_reviews.py
if [ $? -ne 0 ]; then
  echo "Error running fetch_reviews.py. Aborting."
  exit 1
fi

# Step 2: Fetch new Google Play reviews
echo "\n---> Running Google Play fetch script (fetch_google_play_reviews.py)..."
python3 fetch_google_play_reviews.py
if [ $? -ne 0 ]; then
  echo "Error running fetch_google_play_reviews.py. Aborting."
  exit 1
fi

# Step 3: Analyze new reviews and consolidate
echo "\n---> Running analysis script (analyze_reviews.py)..."
python3 analyze_reviews.py
if [ $? -ne 0 ]; then
  echo "Error running analyze_reviews.py. Aborting."
  exit 1
fi

echo "\nReview update process completed successfully."

exit 0 