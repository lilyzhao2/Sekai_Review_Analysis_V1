# analyze_reviews.py
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import sys
import json
import hashlib # Import hashlib for creating unique keys for App Store reviews

# --- Configuration ---
# Input files from the incremental fetch scripts
INPUT_GOOGLE_PLAY = 'sekai_google_play_reviews.csv'
INPUT_APP_STORE = 'sekai_app_store_reviews.csv'
# The single, consolidated output file where analysis is appended
OUTPUT_ANALYZED_ALL = 'sekai_all_reviews_analyzed.csv'

# We'll use Claude 3 Sonnet - it's fast and capable for this task.
# LangChain will automatically use the ANTHROPIC_API_KEY environment variable.
MODEL_NAME = "claude-3-5-sonnet-20240620"
# --- End Configuration ---

# --- LangChain Setup ---

# 1. Define the AI Model
# temperature=0 means we want the model to be as factual and consistent as possible
model = ChatAnthropic(model=MODEL_NAME, temperature=0)

# 2. Define the Prompt Template
# This tells the AI exactly what to do with each review.
# We're asking for specific fields in a JSON format for easy processing later.
prompt_template = """
Analyze the following App Store/Google Play review for the 'Sekai' app, which is focused on AI-powered immersive/interactive storytelling and fanfiction.

Review Text:
"{review_text}"

Based on the review text, please provide the following information in JSON format:

1.  **sentiment**: Classify the overall sentiment ("Positive", "Negative", "Neutral", "Mixed").

2.  **primary_topic**: Identify the single *primary* topic discussed from this list: ["General Feedback", "Roleplay/Interaction", "Story/Narrative", "Creation Tools", "UI/UX", "Performance/Bugs", "Community", "Monetization/Pricing", "Onboarding", "Other"].

3.  **sub_topics**: Identify *all* relevant sub-topics discussed. Choose from the following based on the primary_topic:
    *   If Roleplay/Interaction: ["AI Quality/Repetitiveness", "Context Memory/Continuity", "Input Method/Friction", "Character Consistency", "Multiplayer Experience", "Search/Discovery"].
    *   If Story/Narrative: ["Story Quality/Coherence", "Completeness (Beginning/Middle/End)", "Following Storylines", "Genre Variety", "Lore/World Depth", "Joining Mid-Story"].
    *   If Creation Tools: ["World Building Ease", "Character Creation", "AI Writing Assistance", "Template Usage", "Import/Export", "Publishing Flow", "Mobile Creation Limits"].
    *   If UI/UX: ["Navigation/Layout Clarity", "Visual Design/Aesthetics", "Tab Organization (Home/For You/Interact)", "Information Overload", "Readability"].
    *   If Performance/Bugs: ["Crashes", "Lag/Slow Speed", "Login Issues", "Update Problems"].
    *   If other primary_topic: ["None"].
    *   Add any other distinct sub-topic mentioned if not covered.
    *   If none apply, use ["None"].

4.  **pain_points**: If sentiment is Negative or Mixed, list the specific problems, frictions, or frustrations mentioned by the user, ideally linking them to the sub-topics. Be specific. If Positive/Neutral or no specific pains mentioned, use ["None"].

5.  **positive_points**: If sentiment is Positive or Mixed, list the specific aspects the user liked or praised. Be specific. If Negative/Neutral or no specific praise mentioned, use ["None"].

6.  **mentioned_ui_elements**: List any specific UI elements *explicitly named* (e.g., "character editor", "story feed", "profile page", "dialogue box", "create button"). If none explicitly named, use ["None"].

7.  **feature_request**: If the user suggests a new feature or improvement, describe it briefly. If not, state "None".

8.  **summary**: Provide a brief one-sentence summary of the review's core message.

Return ONLY the JSON object, enclosed in triple backticks (```json ... ```).
Ensure all fields are present.
Ensure fields that should be lists (sub_topics, pain_points, positive_points, mentioned_ui_elements) are *always* JSON lists, even if empty or containing "None".

Example JSON output:
```json
{{
  "sentiment": "Mixed",
  "primary_topic": "Roleplay/Interaction",
  "sub_topics": ["AI Quality/Repetitiveness", "Context Memory/Continuity"],
  "pain_points": ["The AI keeps forgetting previous conversations.", "Responses sometimes feel generic."],
  "positive_points": ["Love the potential for deep roleplay."],
  "mentioned_ui_elements": ["dialogue box"],
  "feature_request": "Add ability to edit AI responses.",
  "summary": "User enjoys the roleplay concept but finds the AI inconsistent and forgetful, suggesting an edit feature."
}}
```

Now, analyze the provided review.

JSON Output:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# 3. Define the Output Parser
# We expect a string output from the model (which should contain the JSON)
parser = StrOutputParser()

# 4. Create the Chain
# This links the prompt, model, and parser together.
chain = prompt | model | parser

# --- Helper Functions ---
def create_app_store_key(row):
    """Creates a reasonably unique key for App Store reviews based on content."""
    # Combine key fields into a single string
    key_string = f"{row.get('userName', '')}_{row.get('date', '')}_{row.get('rating', '')}_{row.get('review', '')}"
    # Create a SHA256 hash
    return "as_" + hashlib.sha256(key_string.encode('utf-8')).hexdigest()

def analyze_review(review_text):
    """Sends a single review to the LangChain chain and parses the JSON output."""
    if not review_text or not isinstance(review_text, str) or len(review_text.strip()) == 0:
        print("Skipping empty or invalid review.")
        # Return structure matching the new prompt
        return {
            "sentiment": "N/A", "primary_topic": "Invalid Input", "sub_topics": [],
            "pain_points": ["Review text was empty or invalid."], "positive_points": [],
            "mentioned_ui_elements": [], "feature_request": "N/A", "summary": "N/A"
        }
    try:
        print(f"\nAnalyzing review: \"{review_text[:100]}...\"")
        raw_result = chain.invoke({"review_text": review_text})
        # print(f"Raw response: {raw_result}")

        json_string = raw_result.strip().lstrip('```json').rstrip('```').strip()
        analysis = json.loads(json_string)

        # Ensure list format for specific fields & presence of all fields
        list_fields = ['sub_topics', 'pain_points', 'positive_points', 'mentioned_ui_elements']
        all_expected_fields = ['sentiment', 'primary_topic', 'sub_topics', 'pain_points', 'positive_points', 'mentioned_ui_elements', 'feature_request', 'summary']
        
        processed_analysis = {}
        for field in all_expected_fields:
            if field not in analysis:
                 # Set default value based on type expectation
                 processed_analysis[field] = [] if field in list_fields else "Missing"
            elif field in list_fields:
                 if not isinstance(analysis[field], list):
                     # Attempt to convert to list or wrap if not list
                     processed_analysis[field] = [str(analysis[field])] if analysis[field] not in [None, "", "None"] else []
                 else:
                    processed_analysis[field] = analysis[field]
            else:
                processed_analysis[field] = analysis[field]
        
        # print(f"Processed analysis: {processed_analysis}")
        return processed_analysis
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}\nRaw response was: {raw_result}")
        return {
            "sentiment": "Error", "primary_topic": "Error", "sub_topics": [],
            "pain_points": [f"Failed to parse LLM JSON response: {e}"], "positive_points": [],
            "mentioned_ui_elements": [], "feature_request": "Error", "summary": "Error processing review."
        }
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
        return {
            "sentiment": "Error", "primary_topic": "Error", "sub_topics": [],
            "pain_points": [f"Unexpected analysis error: {e}"], "positive_points": [],
            "mentioned_ui_elements": [], "feature_request": "Error", "summary": "Error processing review."
        }

# --- Main Incremental Analysis Logic ---

if __name__ == "__main__":
    all_raw_reviews = []

    # 1. Load Google Play Reviews
    try:
        if os.path.exists(INPUT_GOOGLE_PLAY):
            df_gp = pd.read_csv(INPUT_GOOGLE_PLAY)
            df_gp['platform'] = 'Google Play'
            df_gp['unique_review_key'] = 'gp_' + df_gp['reviewId'].astype(str)
            if 'review' not in df_gp.columns: df_gp['review'] = ''
            df_gp['review'] = df_gp['review'].fillna('').astype(str)
            if 'at' in df_gp.columns:
                df_gp.rename(columns={'at': 'review_date'}, inplace=True)
            all_raw_reviews.append(df_gp)
            print(f"Loaded {len(df_gp)} reviews from {INPUT_GOOGLE_PLAY}.")
        else:
            print(f"Warning: {INPUT_GOOGLE_PLAY} not found. Skipping Google Play reviews.")
    except Exception as e:
        print(f"Error loading Google Play reviews from {INPUT_GOOGLE_PLAY}: {e}")

    # 2. Load App Store Reviews
    try:
        if os.path.exists(INPUT_APP_STORE):
            df_as = pd.read_csv(INPUT_APP_STORE)
            df_as['platform'] = 'App Store'
            if 'review' not in df_as.columns: df_as['review'] = ''
            df_as['review'] = df_as['review'].fillna('').astype(str)
            if 'date' in df_as.columns:
                 df_as.rename(columns={'date': 'review_date'}, inplace=True)
            df_as['unique_review_key'] = df_as.apply(create_app_store_key, axis=1)
            all_raw_reviews.append(df_as)
            print(f"Loaded {len(df_as)} reviews from {INPUT_APP_STORE}.")
        else:
            print(f"Warning: {INPUT_APP_STORE} not found. Skipping App Store reviews.")
    except Exception as e:
        print(f"Error loading App Store reviews from {INPUT_APP_STORE}: {e}")

    if not all_raw_reviews:
        print("No raw reviews loaded. Exiting.")
        sys.exit()

    # 3. Combine Raw Reviews
    df_raw_combined = pd.concat(all_raw_reviews, ignore_index=True)
    df_raw_combined = df_raw_combined[df_raw_combined['review'].str.strip() != '']
    print(f"Combined total raw reviews (with text): {len(df_raw_combined)}.")

    # 4. Load Existing Analyzed Review Keys
    existing_keys = set()
    existing_analyzed_df = pd.DataFrame()
    print("*** WARNING: Existing key check disabled - forcing re-analysis of all raw reviews. ***")
    # --- TEMPORARILY DISABLED TO FORCE RE-ANALYSIS ---
    # if os.path.exists(OUTPUT_ANALYZED_ALL):
    #     try:
    #         print(f"Loading existing analyzed keys from {OUTPUT_ANALYZED_ALL}...")
    #         existing_analyzed_df = pd.read_csv(OUTPUT_ANALYZED_ALL, usecols=['unique_review_key'])
    #         existing_keys = set(existing_analyzed_df['unique_review_key'].astype(str))
    #         print(f"Found {len(existing_keys)} existing analyzed review keys.")
    #     except FileNotFoundError:
    #          print(f"{OUTPUT_ANALYZED_ALL} not found. Will create a new file.")
    #     except Exception as e:
    #         print(f"Error loading existing analyzed keys from {OUTPUT_ANALYZED_ALL}: {e}. Proceeding as if file is new.")
    #         existing_keys = set()
    # else:
    #     print(f"{OUTPUT_ANALYZED_ALL} not found. Will create a new file.")
    # --- END TEMPORARY DISABLE ---

    # 5. Identify New Reviews to Analyze
    # --- TEMPORARILY DISABLED - analyze all combined raw reviews instead ---
    # df_new_to_analyze = df_raw_combined[~df_raw_combined['unique_review_key'].isin(existing_keys)].copy()
    df_new_to_analyze = df_raw_combined.copy() # Analyze everything loaded
    # --- END TEMPORARY DISABLE ---
    new_count = len(df_new_to_analyze)

    if new_count == 0:
        # This shouldn't happen now unless raw files are empty
        print("No raw reviews loaded to analyze.") 
        sys.exit()

    print(f"Found {new_count} reviews in raw files to analyze (forcing re-analysis)...")

    # 6. Analyze Only New Reviews
    print("Starting analysis of new reviews...")
    analysis_results = df_new_to_analyze['review'].apply(analyze_review)
    df_analysis = pd.json_normalize(analysis_results)

    df_new_to_analyze.reset_index(drop=True, inplace=True)
    df_analysis.reset_index(drop=True, inplace=True)

    df_new_analyzed = pd.concat([df_new_to_analyze, df_analysis], axis=1)
    print(f"Analysis complete for {len(df_new_analyzed)} new reviews.")

    # 7. Append Results to Consolidated File
    # --- TEMPORARILY CHANGE to OVERWRITE instead of append ---
    print(f"Saving {len(df_new_analyzed)} re-analyzed reviews to {OUTPUT_ANALYZED_ALL} (OVERWRITING)...")
    # write_header = not os.path.exists(OUTPUT_ANALYZED_ALL) or existing_keys == set()
    # save_mode = 'a' if not write_header else 'w'
    write_header = True
    save_mode = 'w' # Force overwrite
    # --- END TEMPORARY CHANGE ---
    
    # Adjust column alignment logic slightly to handle potential missing columns in older files
    # (This part remains useful even when overwriting to ensure consistent column order)
    if save_mode == 'a' and os.path.exists(OUTPUT_ANALYZED_ALL):
        # ... (column alignment logic - less critical when overwriting but keep for consistency) ...
        pass # Keep existing alignment logic just in case, although mode is 'w' now
    else:
        df_to_save = df_new_analyzed

    try:
        df_to_save.to_csv(OUTPUT_ANALYZED_ALL, mode=save_mode, header=write_header, index=False, encoding='utf-8-sig')
        print(f"Successfully saved results to {OUTPUT_ANALYZED_ALL} (Overwritten).")
        print("\nPreview of newly analyzed and appended data (selected columns):")
        preview_cols = ['unique_review_key', 'platform', 'sentiment', 'primary_topic', 'sub_topics', 'pain_points', 'positive_points', 'summary']
        print(df_to_save[[col for col in preview_cols if col in df_to_save.columns]].head().to_string())
    except Exception as e:
        print(f"Error saving results to {OUTPUT_ANALYZED_ALL}: {e}") 