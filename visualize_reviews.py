# visualize_reviews.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import ast # For safely evaluating string representations of lists

# --- Configuration ---
INPUT_FILE = 'sekai_all_reviews_analyzed.csv'
OUTPUT_DIR = '.' # Save plots in the current directory
# --- End Configuration ---

def safe_literal_eval(val):
    """Safely evaluate a string literal (like a list) or return an empty list."""
    if pd.isna(val):
        return []
    try:
        # Attempt to evaluate the string as a Python literal
        evaluated = ast.literal_eval(str(val))
        # Ensure it's a list, otherwise wrap it in a list
        return evaluated if isinstance(evaluated, list) else [evaluated]
    except (ValueError, SyntaxError):
        # If it's not a valid literal (e.g., just a plain string like "None"), return it as a single-item list
        return [str(val)] 

print(f"Loading data from {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_FILE}. Please ensure it exists.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

print(f"Loaded {len(df)} rows.")

# --- Data Cleaning ---
# Filter out rows where sentiment analysis failed
initial_rows = len(df)
df_cleaned = df[df['sentiment'] != 'Error'].copy()
removed_rows = initial_rows - len(df_cleaned)
if removed_rows > 0:
    print(f"Removed {removed_rows} rows with 'Error' sentiment.")

if df_cleaned.empty:
    print("No valid data remaining after filtering errors. Cannot generate plots.")
    sys.exit(0)

# Convert review_date to datetime objects, coercing errors to NaT (Not a Time)
df_cleaned['review_date'] = pd.to_datetime(df_cleaned['review_date'], errors='coerce')

# Drop rows where date conversion failed, if any
initial_rows = len(df_cleaned)
df_cleaned.dropna(subset=['review_date'], inplace=True)
removed_rows = initial_rows - len(df_cleaned)
if removed_rows > 0:
    print(f"Removed {removed_rows} rows with invalid dates.")
    
if df_cleaned.empty:
    print("No valid data remaining after cleaning dates. Cannot generate plots.")
    sys.exit(0)
    
print(f"Data cleaned. {len(df_cleaned)} rows remaining for analysis.")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Plot Generation ---
print("Generating plots...")

# 1. Overall Sentiment Distribution
plt.figure(figsize=(8, 5))
sentiment_counts = df_cleaned['sentiment'].value_counts()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.title('Overall Sentiment Distribution')
plt.ylabel('Number of Reviews')
plt.xlabel('Sentiment')
plt.tight_layout()
sentiment_plot_path = os.path.join(OUTPUT_DIR, 'sentiment_distribution.png')
plt.savefig(sentiment_plot_path)
plt.close() # Close the plot to free memory
print(f" - Saved: {sentiment_plot_path}")

# 2. Sentiment Distribution by Platform
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='sentiment', hue='platform', palette="magma", order=sentiment_counts.index)
plt.title('Sentiment Distribution by Platform')
plt.ylabel('Number of Reviews')
plt.xlabel('Sentiment')
plt.tight_layout()
platform_sentiment_plot_path = os.path.join(OUTPUT_DIR, 'sentiment_by_platform.png')
plt.savefig(platform_sentiment_plot_path)
plt.close()
print(f" - Saved: {platform_sentiment_plot_path}")

# 3. Sentiment Trend Over Time (Monthly)
# Set date as index for time-series analysis
df_time_indexed = df_cleaned.set_index('review_date')
# Resample by month ('M'), count sentiments, unstack for plotting, fill missing months with 0
sentiment_over_time = df_time_indexed.groupby(pd.Grouper(freq='M'))['sentiment'].value_counts().unstack().fillna(0)

plt.figure(figsize=(14, 7))
sentiment_over_time.plot(kind='line', ax=plt.gca()) # Use gca() to plot on the current figure axes
plt.title('Sentiment Trend Over Time (Monthly)')
plt.ylabel('Number of Reviews')
plt.xlabel('Month')
plt.legend(title='Sentiment')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
time_sentiment_plot_path = os.path.join(OUTPUT_DIR, 'sentiment_over_time.png')
plt.savefig(time_sentiment_plot_path)
plt.close()
print(f" - Saved: {time_sentiment_plot_path}")

# 4. Top Pain Points (from Negative/Mixed Reviews)
print("Analyzing pain points...")
df_negative = df_cleaned[df_cleaned['sentiment'].isin(['Negative', 'Mixed'])].copy()

if not df_negative.empty and 'pain_points' in df_negative.columns:
    # Apply safe parsing and explode
    df_negative['pain_points_list'] = df_negative['pain_points'].apply(safe_literal_eval)
    df_pain_points = df_negative.explode('pain_points_list')
    
    # Filter out common non-informative entries
    df_pain_points = df_pain_points[~df_pain_points['pain_points_list'].astype(str).str.lower().isin(['none', '', 'nan'])]
    df_pain_points = df_pain_points.dropna(subset=['pain_points_list'])

    if not df_pain_points.empty:
        top_pain_points = df_pain_points['pain_points_list'].value_counts().head(15) # Get top 15
        
        if not top_pain_points.empty:
            plt.figure(figsize=(10, 8))
            sns.barplot(x=top_pain_points.values, y=top_pain_points.index, palette="rocket")
            plt.title('Top 15 Pain Points in Negative/Mixed Reviews')
            plt.xlabel('Frequency')
            plt.ylabel('Pain Point')
            plt.tight_layout()
            pain_points_plot_path = os.path.join(OUTPUT_DIR, 'top_pain_points.png')
            plt.savefig(pain_points_plot_path)
            plt.close()
            print(f" - Saved: {pain_points_plot_path}")
        else:
            print(" - No significant pain points found after filtering.")
    else:
        print(" - No pain points data to analyze after filtering.")
else:
    print(" - No negative/mixed reviews found or 'pain_points' column missing.")

print("\nVisualization script finished.")
print(f"Plots saved in: {os.path.abspath(OUTPUT_DIR)}") 