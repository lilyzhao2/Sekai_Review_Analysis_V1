# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import ast # For safely evaluating string representations of lists
from datetime import datetime
import numpy as np

# --- Add LangChain imports for summaries ---
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os # To potentially check for API key env var

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Sekai Review Dashboard")
st.title("Sekai App Review Analysis Dashboard")

# --- Configuration ---
INPUT_FILE = 'sekai_all_reviews_analyzed.csv'

# Columns expected to contain list-like strings
LIST_COLUMNS = ['sub_topics', 'pain_points', 'positive_points', 'mentioned_ui_elements']

# --- Helper Functions ---
def safe_literal_eval(val):
    """Safely evaluate a string literal (like a list) or return an empty list."""
    if pd.isna(val):
        return []
    try:
        evaluated = ast.literal_eval(str(val))
        # Ensure the result is always a list
        if isinstance(evaluated, list):
            # Filter out any potential "None" strings or empty strings within the list
            return [item for item in evaluated if item and str(item).strip().lower() not in ['none', '']]
        elif evaluated and str(evaluated).strip().lower() not in ['none', '']:
             return [evaluated] # Wrap single items in a list
        else:
            return [] # Return empty list if evaluation resulted in None/empty string
    except (ValueError, SyntaxError):
        # If literal_eval fails, treat the original string as a single-item list if it's not None/empty
        clean_val = str(val).strip()
        return [clean_val] if clean_val and clean_val.lower() not in ['none', ''] else []

# --- LangChain Setup for Summaries & Analysis ---
@st.cache_resource # Cache the model connection
def get_llm_model():
    # Define model name inside the function to avoid scope issues with caching
    model_name = "claude-3-5-sonnet-20240620"
    try:
        if not os.getenv("ANTHROPIC_API_KEY"):
             st.warning("ANTHROPIC_API_KEY environment variable not set. AI analysis features will be disabled.", icon="⚠️")
             return None
        return ChatAnthropic(model=model_name, temperature=0.2, max_retries=1)
    except Exception as e:
        st.error(f"Failed to initialize Anthropic model: {e}")
        return None

lm = get_llm_model() # Use a shorter name for the language model instance

# --- Setup for Chart Summaries ---
if lm:
    summary_parser = StrOutputParser()
    summary_prompt_template = ChatPromptTemplate.from_template(
        """
        Analyze the following data which is being used to generate a chart titled '{chart_description}'.
        The data represents counts or frequencies based on filtered user reviews for the Sekai app.
        
        Data:
        {data_string}
        
        Provide a concise 1-2 sentence summary highlighting the key insights, dominant categories, or notable distributions based *only* on this provided data. Focus on the most important takeaways a product manager would want to know quickly from this chart data.
        Be objective and data-driven in your summary.
        Summary:
        """
    )
    summary_chain = summary_prompt_template | lm | summary_parser
else:
    summary_chain = None

# --- Setup for Feature Request Analysis ---
if lm:
    feature_analysis_parser = StrOutputParser()
    feature_analysis_prompt_template = ChatPromptTemplate.from_template(
         """
        You are analyzing a list of feature requests extracted from user reviews for the Sekai app (an AI storytelling platform with ~50k DAU). Your goal is to group similar requests and provide a qualitative analysis inspired by the RICE framework based *only* on the text provided.

        RICE Components (for your context, not for numerical output):
        *   Reach: How many users might this affect? (Estimate qualitatively based on frequency/language)
        *   Impact: How significant would this feature be to users? (Infer from user language/tone)
        *   Confidence: How sure are we about the reach/impact based *only* on this list? (Frequency in list)
        *   Effort: How much work is involved? (State this cannot be determined from reviews).

        Here is a list of extracted feature requests (and review summaries for context):
        {feature_requests_string}

        Based *only* on the provided list:
        1.  Identify the top 3-5 recurring themes or categories of feature requests.
        2.  For each theme, provide a brief qualitative assessment:
            *   **Theme:** (e.g., Improved Character Consistency)
            *   **Potential Reach:** (e.g., Seems frequently mentioned, suggesting moderate reach; Mentioned occasionally)
            *   **Potential Impact:** (e.g., Users describe this as highly frustrating/critical, suggesting high impact; Seems like a quality-of-life improvement, suggesting medium impact)
            *   **Confidence (from list frequency):** (e.g., High - appears many times in this list; Medium - appears several times; Low - appears once or twice)
            *   **Effort:** Cannot be determined from reviews; requires team input.
        3.  Provide a 1-2 sentence concluding summary identifying the most prominent request themes emerging from this specific list.

        Structure your response clearly. Focus only on information present in the request list.

        Analysis:
        """
    )
    feature_analysis_chain = feature_analysis_prompt_template | lm | feature_analysis_parser
else:
    feature_analysis_chain = None

# --- Cached Functions ---
@st.cache_data(max_entries=20)
def generate_chart_summary(chart_data_series, chart_description):
    """Generates a summary for chart data using LangChain if the model is available."""
    if summary_chain is None or chart_data_series is None or chart_data_series.empty:
        return "*Summary generation disabled or no data available.*"
        
    # Prepare data string (limit length if necessary)
    # Convert Series/DataFrame to string, handle potential large data
    if isinstance(chart_data_series, pd.Series):
        data_string = chart_data_series.head(15).to_string() # Limit for prompt
    elif isinstance(chart_data_series, pd.DataFrame):
         data_string = chart_data_series.head(15).to_string() # Limit for prompt
    else:
        data_string = str(chart_data_series)
        
    if len(data_string) > 1500: # Add length limit safeguard
        data_string = data_string[:1500] + "... (truncated)"
        
    try:
        print(f"Requesting summary for: {chart_description}") # Log when actually calling API
        summary = summary_chain.invoke({"chart_description": chart_description, "data_string": data_string})
        return summary
    except Exception as e:
        # Log the error but return a user-friendly message
        print(f"Error generating summary for {chart_description}: {e}")
        # Check for specific error messages if needed (e.g., API key, credits)
        error_message = str(e)
        user_facing_error = "Could not generate summary due to an API error."
        if "credit balance" in error_message.lower():
             user_facing_error = "Could not generate summary: Check API credit balance."
        elif "api key" in error_message.lower():
             user_facing_error = "Could not generate summary: Check API key configuration."
        return f"*{user_facing_error}*"

@st.cache_data(max_entries=10)
def generate_feature_request_analysis(feature_requests_df):
    """Generates a qualitative analysis of feature requests using LangChain."""
    if feature_analysis_chain is None or feature_requests_df is None or feature_requests_df.empty:
        return "*Feature request analysis disabled or no requests found.*"

    # Prepare data string (limit to relevant columns and rows)
    try:
        # Focus on the request and maybe summary, limit length
        data_string = feature_requests_df[['feature_request', 'summary']].head(25).to_string(index=False)
        if len(data_string) > 2000: # Limit prompt size
            data_string = data_string[:2000] + "... (truncated list)"
    except Exception as e:
        return f"*Error preparing feature request data for analysis: {e}*"

    try:
        print("Requesting feature request analysis...")
        analysis_text = feature_analysis_chain.invoke({"feature_requests_string": data_string})
        return analysis_text
    except Exception as e:
        print(f"Error generating feature request analysis: {e}")
        error_message = str(e)
        user_facing_error = "Could not generate feature analysis due to an API error."
        if "credit balance" in error_message.lower():
             user_facing_error = "Could not generate analysis: Check API credit balance."
        elif "api key" in error_message.lower():
             user_facing_error = "Could not generate analysis: Check API key configuration."
        return f"*{user_facing_error}*"

# --- Load and Cache Data ---
@st.cache_data
def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, low_memory=False) # Added low_memory=False for potential mixed types
        print(f"Loaded {len(df)} rows.")

        # --- Initial Cleaning ---
        initial_rows = len(df)
        # Handle potential analysis errors - check if 'sentiment' column exists first
        if 'sentiment' in df.columns:
            df_cleaned = df[df['sentiment'] != 'Error'].copy()
            removed_rows = initial_rows - len(df_cleaned)
            if removed_rows > 0:
                print(f"Removed {removed_rows} rows with 'Error' sentiment.")
        else:
             print("Warning: 'sentiment' column not found. Skipping error filtering.")
             df_cleaned = df.copy()

        if df_cleaned.empty:
            print("No valid data after initial filtering.")
            return pd.DataFrame()

        # Clean date column
        if 'review_date' in df_cleaned.columns:
            df_cleaned['review_date'] = pd.to_datetime(df_cleaned['review_date'], errors='coerce')
            initial_rows_date = len(df_cleaned)
            df_cleaned.dropna(subset=['review_date'], inplace=True)
            removed_rows_date = initial_rows_date - len(df_cleaned)
            if removed_rows_date > 0:
                print(f"Removed {removed_rows_date} rows with invalid dates.")
        else:
             print("Warning: 'review_date' column not found. Cannot filter by date.")
             # Add a placeholder date if missing? Or handle downstream filtering errors.
             # For now, let it proceed, filtering might fail later.

        # Convert list-like string columns to actual lists
        print("Applying safe_literal_eval to list columns...")
        for col in LIST_COLUMNS:
            if col in df_cleaned.columns:
                list_col_name = f"{col}_list"
                print(f"  Processing column: {col} -> {list_col_name}")
                df_cleaned[list_col_name] = df_cleaned[col].apply(safe_literal_eval)
            else:
                print(f"  Warning: Expected list column '{col}' not found.")

        # --- Infer User Segment --- 
        print("Inferring user segments...")
        if 'primary_topic' in df_cleaned.columns:
            conditions = [
                df_cleaned['primary_topic'] == 'Creation Tools',
                df_cleaned['primary_topic'] == 'Roleplay/Interaction',
                df_cleaned['primary_topic'] == 'Story/Narrative'
            ]
            choices = ['Creator', 'Roleplayer', 'Passive Viewer']
            # Use numpy.select for efficient conditional assignment
            df_cleaned['inferred_user_segment'] = np.select(conditions, choices, default='Other/General')
            print(f"Inferred segments breakdown:\n{df_cleaned['inferred_user_segment'].value_counts()}")
        else:
            print("Warning: 'primary_topic' column not found. Cannot infer user segments.")
            df_cleaned['inferred_user_segment'] = 'Unknown' # Add placeholder column

        print(f"Data loading and cleaning complete. {len(df_cleaned)} rows remaining.")
        return df_cleaned

    except FileNotFoundError:
        st.error(f"Error: Input file not found at {INPUT_FILE}. Please ensure it exists and the pipeline has run.")
        return None
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.exception(e) # Show full traceback in Streamlit app for debugging
        return None

# Load the full dataset
df_full = load_data(INPUT_FILE)

# --- Sidebar Filters ---
st.sidebar.header("Filters")

if df_full is not None and not df_full.empty:
    # Platform Filter
    if 'platform' in df_full.columns:
        platforms = sorted(df_full['platform'].unique().tolist())
        selected_platforms = st.sidebar.multiselect("Platform(s):", options=platforms, default=platforms)
    else:
        st.sidebar.warning("'platform' column missing. Cannot filter by platform.")
        selected_platforms = []

    # Date Range Filter
    if 'review_date' in df_full.columns:
        min_date = df_full['review_date'].min().date()
        max_date = df_full['review_date'].max().date()
        selected_date_range = st.sidebar.date_input("Date Range:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        st.sidebar.warning("'review_date' column missing. Cannot filter by date.")
        selected_date_range = None
        
    # Sentiment Filter
    if 'sentiment' in df_full.columns:
        sentiments = sorted(df_full['sentiment'].unique().tolist())
        selected_sentiments = st.sidebar.multiselect("Sentiment(s):", options=sentiments, default=sentiments)
    else:
        st.sidebar.warning("'sentiment' column missing. Cannot filter by sentiment.")
        selected_sentiments = []
        
    # Primary Topic Filter
    if 'primary_topic' in df_full.columns:
        primary_topics = sorted(df_full['primary_topic'].dropna().unique().tolist())
        selected_primary_topics = st.sidebar.multiselect("Primary Topic(s):", options=primary_topics, default=primary_topics)
    else:
         st.sidebar.warning("'primary_topic' column missing. Cannot filter by primary topic.")
         selected_primary_topics = []

    # --- NEW: Inferred User Segment Filter ---
    if 'inferred_user_segment' in df_full.columns:
        segments = sorted(df_full['inferred_user_segment'].unique().tolist())
        selected_segments = st.sidebar.multiselect("Inferred User Segment(s):", options=segments, default=segments)
    else:
        st.sidebar.warning("'inferred_user_segment' column missing. Cannot filter by segment.")
        selected_segments = []

    # Apply Filters
    df_filtered = df_full.copy()
    if 'platform' in df_filtered.columns and selected_platforms:
        df_filtered = df_filtered[df_filtered['platform'].isin(selected_platforms)]
    if 'review_date' in df_filtered.columns and selected_date_range and len(selected_date_range) == 2:
        start_date = pd.to_datetime(selected_date_range[0])
        end_date = pd.to_datetime(selected_date_range[1])
        df_filtered = df_filtered[(df_filtered['review_date'] >= start_date) & (df_filtered['review_date'] <= end_date)]
    if 'sentiment' in df_filtered.columns and selected_sentiments:
         df_filtered = df_filtered[df_filtered['sentiment'].isin(selected_sentiments)]
    if 'primary_topic' in df_filtered.columns and selected_primary_topics:
         df_filtered = df_filtered[df_filtered['primary_topic'].isin(selected_primary_topics)]
    # --- Add segment filter application ---
    if 'inferred_user_segment' in df_filtered.columns and selected_segments:
         df_filtered = df_filtered[df_filtered['inferred_user_segment'].isin(selected_segments)]
         
    st.sidebar.info(f"Displaying {len(df_filtered)} reviews based on filters.")

else:
    st.sidebar.error("Could not load data for filtering.")
    df_filtered = pd.DataFrame()

# --- Main Dashboard Area ---
if df_filtered is not None and not df_filtered.empty:

    # --- Key Metrics Row ---
    st.subheader("Overall Metrics (Filtered)")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total Reviews", len(df_filtered))
    avg_rating = df_filtered['rating'].mean() if 'rating' in df_filtered.columns else 'N/A'
    m_col2.metric("Average Rating", f"{avg_rating:.2f}" if isinstance(avg_rating, (int, float)) else avg_rating)
    # Calculate dominant sentiment
    if 'sentiment' in df_filtered.columns and not df_filtered['sentiment'].empty:
        dominant_sentiment = df_filtered['sentiment'].mode()[0]
    else:
        dominant_sentiment = "N/A"
    m_col3.metric("Dominant Sentiment", dominant_sentiment)

    st.divider()

    # --- Sentiment Analysis Section ---
    st.header("Sentiment & Topic Analysis")
    # Add option to group charts by segment
    group_by_segment = st.checkbox("Group charts by User Segment?", key='segment_grouping', value=False)
    hue_param = 'inferred_user_segment' if group_by_segment else None
    
    s_col1, s_col2 = st.columns(2)

    with s_col1:
        st.subheader("Sentiment Distribution")
        if 'sentiment' in df_filtered.columns:
            sentiment_counts = df_filtered['sentiment'].value_counts()
            if not sentiment_counts.empty:
                fig1, ax1 = plt.subplots(figsize=(7, 5))
                sns.countplot(data=df_filtered, x='sentiment', hue=hue_param, palette="viridis", ax=ax1, order=sentiment_counts.index)
                ax1.set_title(f'Sentiment Distribution {("by Segment" if group_by_segment else "")}')
                ax1.set_ylabel('Review Count')
                ax1.set_xlabel('Sentiment')
                plt.xticks(rotation=45, ha='right')
                if group_by_segment:
                    ax1.legend(title='User Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig1)
                # --- Generate and display summary ---
                summary_text = generate_chart_summary(sentiment_counts, f"Sentiment Distribution{(' grouped by User Segment' if group_by_segment else '')}")
                st.markdown(f"**AI Summary:** {summary_text}")
            else:
                st.info("No sentiment data for current filters.")
        else: 
             st.warning("'sentiment' column missing.")

    with s_col2:
        st.subheader("Primary Topic Distribution")
        if 'primary_topic' in df_filtered.columns:
            topic_counts = df_filtered['primary_topic'].value_counts()
            if not topic_counts.empty:
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                sns.countplot(data=df_filtered, x='primary_topic', hue=hue_param, palette="magma", ax=ax2, order=topic_counts.index)
                ax2.set_title(f'Primary Topic Distribution {("by Segment" if group_by_segment else "")}')
                ax2.set_ylabel('Review Count')
                ax2.set_xlabel('Primary Topic')
                plt.xticks(rotation=45, ha='right')
                if group_by_segment:
                     ax2.legend(title='User Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig2)
                # --- Generate and display summary ---
                summary_text = generate_chart_summary(topic_counts, f"Primary Topic Distribution{(' grouped by User Segment' if group_by_segment else '')}")
                st.markdown(f"**AI Summary:** {summary_text}")
            else:
                st.info("No primary topic data for current filters.")
        else:
            st.warning("'primary_topic' column missing.")
            
    # Sub-topic Analysis
    st.subheader("Sub-Topic Frequency")
    if 'sub_topics_list' in df_filtered.columns:
        df_sub_topics = df_filtered.explode('sub_topics_list').dropna(subset=['sub_topics_list'])
        df_sub_topics = df_sub_topics[df_sub_topics['sub_topics_list'] != '']
        if not df_sub_topics.empty:
            top_n_sub = st.slider("Number of Top Sub-Topics:", min_value=5, max_value=30, value=15, key='subtopics_slider')
            sub_topic_counts = df_sub_topics['sub_topics_list'].value_counts().head(top_n_sub)
            if not sub_topic_counts.empty:
                fig_st, ax_st = plt.subplots(figsize=(10, max(6, top_n_sub * 0.35)))
                sns.barplot(x=sub_topic_counts.values, y=sub_topic_counts.index, palette="crest", ax=ax_st)
                ax_st.set_title(f'Top {top_n_sub} Sub-Topics')
                ax_st.set_xlabel('Frequency')
                ax_st.set_ylabel('Sub-Topic')
                plt.tight_layout()
                st.pyplot(fig_st)
                # --- Generate and display summary ---
                summary_text = generate_chart_summary(sub_topic_counts, f"Top {top_n_sub} Sub-Topics")
                st.markdown(f"**AI Summary:** {summary_text}")
            else:
                st.info("No sub-topics found for the current filters.")
        else:
            st.info("No sub-topics data to analyze for the current filters.")
    else:
        st.warning("'sub_topics_list' column missing.")


    st.divider()

    # --- Trend Analysis Section ---
    st.header("Trend Analysis")
    st.subheader("Sentiment Over Time (Monthly)")
    if 'review_date' in df_filtered.columns and 'sentiment' in df_filtered.columns:
        df_time_indexed = df_filtered.set_index('review_date')
        # Use ME for Month End frequency
        sentiment_over_time = df_time_indexed.groupby(pd.Grouper(freq='ME'))['sentiment'].value_counts().unstack().fillna(0)
        if not sentiment_over_time.empty:
            st.line_chart(sentiment_over_time)
        else:
            st.info("No data available for sentiment trend within the selected filters.")
    else:
        st.warning("'review_date' or 'sentiment' column missing. Cannot plot trend.")

    st.divider()

    # --- Positive vs Negative Feedback ---
    st.header("Detailed Feedback Analysis")
    fb_col1, fb_col2 = st.columns(2)
    
    with fb_col1:
        st.subheader("Top Pain Points")
        if 'pain_points_list' in df_filtered.columns:
            df_pain_points = df_filtered.explode('pain_points_list').dropna(subset=['pain_points_list'])
            df_pain_points = df_pain_points[df_pain_points['pain_points_list'] != '']
            if not df_pain_points.empty:
                top_n_pain = st.slider("Number of Top Pain Points:", min_value=5, max_value=25, value=10, key='painpoints_slider')
                top_pain_points = df_pain_points['pain_points_list'].value_counts().head(top_n_pain)
                if not top_pain_points.empty:
                    fig3, ax3 = plt.subplots(figsize=(10, max(6, top_n_pain * 0.4)))
                    sns.barplot(x=top_pain_points.values, y=top_pain_points.index, palette="rocket", ax=ax3)
                    ax3.set_title(f'Top {top_n_pain} Pain Points')
                    ax3.set_xlabel('Frequency')
                    ax3.set_ylabel('Pain Point')
                    plt.tight_layout()
                    st.pyplot(fig3)
                    # --- Generate and display summary ---
                    summary_text = generate_chart_summary(top_pain_points, f"Top {top_n_pain} Pain Points")
                    st.markdown(f"**AI Summary:** {summary_text}")
                else:
                    st.info("No significant pain points found after filtering.")
            else:
                st.info("No pain points data to analyze for the current filters.")
        else:
            st.warning("'pain_points_list' column missing.")

    with fb_col2:
        st.subheader("Top Positive Points")
        if 'positive_points_list' in df_filtered.columns:
            df_positive = df_filtered.explode('positive_points_list').dropna(subset=['positive_points_list'])
            df_positive = df_positive[df_positive['positive_points_list'] != '']
            if not df_positive.empty:
                top_n_pos = st.slider("Number of Top Positive Points:", min_value=5, max_value=25, value=10, key='positive_slider')
                top_positive_points = df_positive['positive_points_list'].value_counts().head(top_n_pos)
                if not top_positive_points.empty:
                    fig4, ax4 = plt.subplots(figsize=(10, max(6, top_n_pos * 0.4)))
                    sns.barplot(x=top_positive_points.values, y=top_positive_points.index, palette="summer", ax=ax4)
                    ax4.set_title(f'Top {top_n_pos} Positive Points')
                    ax4.set_xlabel('Frequency')
                    ax4.set_ylabel('Positive Point')
                    plt.tight_layout()
                    st.pyplot(fig4)
                    # --- Generate and display summary ---
                    summary_text = generate_chart_summary(top_positive_points, f"Top {top_n_pos} Positive Points")
                    st.markdown(f"**AI Summary:** {summary_text}")
                else:
                    st.info("No significant positive points found after filtering.")
            else:
                st.info("No positive points data to analyze for the current filters.")
        else:
             st.warning("'positive_points_list' column missing.")
             
    st.divider()

    # --- UI Element & Feature Request Section ---
    st.header("UI Elements & Feature Requests")
    ui_col1, ui_col2 = st.columns(2)
    
    with ui_col1:
        st.subheader("Mentioned UI Elements")
        if 'mentioned_ui_elements_list' in df_filtered.columns:
            df_ui = df_filtered.explode('mentioned_ui_elements_list').dropna(subset=['mentioned_ui_elements_list'])
            df_ui = df_ui[df_ui['mentioned_ui_elements_list'] != '']
            if not df_ui.empty:
                 top_n_ui = st.slider("Number of Top UI Elements:", min_value=3, max_value=20, value=10, key='ui_slider')
                 top_ui_elements = df_ui['mentioned_ui_elements_list'].value_counts().head(top_n_ui)
                 if not top_ui_elements.empty:
                     fig5, ax5 = plt.subplots(figsize=(10, max(5, top_n_ui * 0.4)))
                     sns.barplot(x=top_ui_elements.values, y=top_ui_elements.index, palette="coolwarm", ax=ax5)
                     ax5.set_title(f'Top {top_n_ui} Mentioned UI Elements')
                     ax5.set_xlabel('Frequency')
                     ax5.set_ylabel('UI Element')
                     plt.tight_layout()
                     st.pyplot(fig5)
                     # --- Generate and display summary ---
                     summary_text = generate_chart_summary(top_ui_elements, f"Top {top_n_ui} Mentioned UI Elements")
                     st.markdown(f"**AI Summary:** {summary_text}")
                 else:
                     st.info("No specific UI elements mentioned frequently in the filtered data.")
            else:
                 st.info("No mentioned UI element data to analyze.")
        else:
             st.warning("'mentioned_ui_elements_list' column missing.")
             
    with ui_col2:
        st.subheader("Feature Requests")
        if 'feature_request' in df_filtered.columns:
            df_requests = df_filtered[df_filtered['feature_request'].fillna('None').str.strip().str.lower() != 'none']
            if not df_requests.empty:
                 display_cols = ['review_date', 'platform', 'feature_request', 'summary']
                 display_cols = [col for col in display_cols if col in df_requests.columns]
                 rows_to_summarize = df_requests[display_cols].head(25) # Use same subset for table & analysis
                 st.dataframe(rows_to_summarize, height=300) # Display limited rows for clarity
                 
                 st.markdown("--- RICE-Inspired AI Analysis (Experimental) ---")
                 with st.spinner("Analyzing feature requests..."):
                      analysis_text = generate_feature_request_analysis(rows_to_summarize)
                 st.markdown(analysis_text) # Display the qualitative analysis
            else:
                 st.info("No specific feature requests found in the filtered data.")
        else:
            st.warning("'feature_request' column missing.")

    st.divider()
    # Add a section to view raw data table
    st.header("Raw Data Explorer")
    show_data = st.checkbox("Show filtered review data table")
    if show_data:
        # Select a subset of columns for better readability
        cols_to_show = [
            'review_date', 'platform', 'rating', 'sentiment', 'primary_topic',
             'sub_topics', 'pain_points', 'positive_points', 'mentioned_ui_elements',
              'feature_request', 'summary', 'review' # Keep original text too
        ]
        # Filter to only columns that actually exist in df_filtered
        cols_to_show_present = [col for col in cols_to_show if col in df_filtered.columns]
        st.dataframe(df_filtered[cols_to_show_present])


elif df_full is None:
    st.error(f"Dashboard cannot run: Error loading data from {INPUT_FILE}.")

elif df_full.empty:
    st.warning("No review data available to display after initial cleaning.")

# To run this dashboard: `streamlit run dashboard.py` 