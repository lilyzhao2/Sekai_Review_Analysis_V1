
Here's an explanation of the methodology used to create this dashboard and derive the insights:

1.  **Data Acquisition & Preparation:**
    *   **Source:** The process starts by fetching raw user reviews directly from the Google Play Store and Apple App Store using specialized Python libraries (`google-play-scraper`, `app-store-scraper`).
    *   **Incremental Fetching:** The fetching scripts (`fetch_reviews.py`, `fetch_google_play_reviews.py`) are designed to run incrementally. They check the latest review already saved locally (in `sekai_app_store_reviews.csv` and `sekai_google_play_reviews.csv`) and only download reviews newer than that, appending them to the respective files. This avoids re-fetching old data.
    *   **Consolidation (Initial):** The raw reviews from both platforms are stored in separate CSV files initially.

2.  **Core AI Analysis (Batch Processing):**
    *   **Tool:** This is handled by the `analyze_reviews.py` script.
    *   **Engine:** It utilizes the `LangChain` library to interact with a powerful Large Language Model (LLM) – specifically `Claude 3.5 Sonnet` via the Anthropic API.
    *   **Process:**
        *   The script reads both raw review files.
        *   It identifies which reviews haven't been analyzed yet by comparing unique identifiers (`reviewId` for Google Play, a content hash for App Store) against the master analyzed file (`sekai_all_reviews_analyzed.csv`).
        *   For *each new review*, it sends the review text to the Claude model using a carefully crafted prompt.
        *   **Prompting for Structured Data:** The prompt instructs Claude to act as an analyst and extract specific pieces of information in a structured JSON format:
            *   `sentiment`: Overall feeling (Positive, Negative, Neutral, Mixed).
            *   `primary_topic`: The main subject (Roleplay, Creation, UI/UX, etc.).
            *   `sub_topics`: More specific themes within the primary topic (e.g., "AI Quality/Repetitiveness" under Roleplay).
            *   `pain_points`: Specific user complaints (if sentiment is Negative/Mixed).
            *   `positive_points`: Specific aspects the user liked (if sentiment is Positive/Mixed).
            *   `mentioned_ui_elements`: Explicit UI components named by the user.
            *   `feature_request`: Any suggested features or improvements.
            *   `summary`: A one-sentence summary of the review.
        *   **Output:** The script parses the JSON response from Claude and appends these structured analysis results, along with the original review data and platform information, to the consolidated `sekai_all_reviews_analyzed.csv` file.
    *   **Timing:** This AI analysis happens *before* the dashboard is loaded, as part of the data update pipeline (`update_reviews.sh`).

3.  **User Segmentation (Dashboard Loading):**
    *   **Tool:** This logic resides within the `load_data` function in the `dashboard.py` script.
    *   **Methodology:** Since we don't have explicit user type data, we *infer* a likely user segment based on the *content* of their review, specifically the `primary_topic` assigned by the AI in the previous step.
    *   **Rules:**
        *   If `primary_topic` is "Creation Tools", segment is "Creator".
        *   If `primary_topic` is "Roleplay/Interaction", segment is "Roleplayer".
        *   If `primary_topic` is "Story/Narrative", segment is "Passive Viewer".
        *   Otherwise, the segment is "Other/General".
    *   **Nature:** This is a heuristic – an educated guess based on what the user chose to focus on in their review text. It helps group reviews by likely user intent.

4.  **Dashboard Interaction & Visualization:**
    *   **Tool:** The dashboard itself is built using `Streamlit`.
    *   **Loading:** It reads the processed `sekai_all_reviews_analyzed.csv` file.
    *   **Filtering:** It provides interactive sidebar filters allowing you (the user) to slice the data by:
        *   Platform (Android/iOS)
        *   Date Range
        *   Sentiment (AI-assigned)
        *   Primary Topic (AI-assigned)
        *   Inferred User Segment
    *   **Visualization:** Based on the filtered data, it generates various charts (using `Matplotlib` and `Seaborn`) to show distributions (sentiment, topics), frequencies (pain points, positive points, UI elements), and trends (sentiment over time). It also displays feature requests in a table.

5.  **Dynamic AI Summaries & Analysis (Dashboard Interaction):**
    *   **Tool:** This also happens within `dashboard.py`, again using `LangChain` and `Claude 3.5 Sonnet`.
    *   **Process:**
        *   **Chart Summaries:** For each major chart, after the filtered data for *that specific chart* is calculated, this data (counts and percentages) is sent to Claude with a prompt asking for a concise, percentage-focused summary.
        *   **Feature Analysis:** For the feature request table, the displayed list of requests is sent to Claude with a prompt asking for a qualitative RICE-inspired analysis (grouping themes, assessing potential reach/impact based on the text).
        *   **Overall Summary:** An on-demand button compiles key statistics from the *entire currently filtered dataset*, sends them to Claude, and displays a high-level overview.
    *   **Caching:** These AI calls within the dashboard use Streamlit's caching (`@st.cache_data`, `@st.cache_resource`) to store results and avoid redundant API calls if the underlying filtered data hasn't changed, improving performance and reducing cost.

In essence, the methodology uses AI in two phases: first, a structured batch analysis to tag and categorize every review, and second, dynamic AI analysis within the dashboard to summarize and interpret the *filtered results* of the first phase in response to user interaction.


Next Steps for Improvement (Prioritized)
To evolve this dashboard into a robust, reliable internal tool for Sekai:
High Priority (Foundation for Internal Use):
Database Backend (Replace CSVs):
Action: Migrate data storage from CSV files to a SQL database (e.g., PostgreSQL). Update analyze_reviews.py to write to the DB and dashboard.py to query it.
Why: Crucial for performance (especially filtering), scalability, concurrent access reliability, data integrity, and easier integration with other potential tools.
Pipeline Automation & Monitoring:
Action: Schedule the update_reviews.sh script (fetching & analysis) to run automatically (e.g., daily) using cron, GitHub Actions, or a similar task scheduler. Implement robust error handling (try/except blocks around critical steps) and basic notifications (e.g., Slack webhook via curl in the shell script) on failure.
Why: Ensures data freshness without manual effort and provides visibility into pipeline health.
Deployment & Authentication:
Action: Deploy the Streamlit app to a persistent web server (internal server, Streamlit Community Cloud, AWS/GCP/Azure
service). Implement access control (e.g., password protection, company SSO integration) so only authorized Sekai employees can view it.
Why: Makes the dashboard easily accessible to the team while protecting potentially sensitive review data/analysis.
Restore Incremental Analysis:
Action: Revert the temporary changes made to analyze_reviews.py (re-enable existing key checking, change save mode back to 'a'ppend).
Why: Essential for efficient and cost-effective future updates; avoids unnecessary re-processing of old reviews and high API costs.
Medium Priority (Improving Maintainability & Data Context):
Configuration Management:
Action: Move API keys, database connection details (once migrated), file paths, and model names into environment variables or a configuration file (.env, config.yaml). Ensure API keys are securely managed (not in code/git).
Why: Improves security and makes configuration easier across different environments (local vs. deployment).
Code Quality & Documentation:
Action: Create requirements.txt (pip3 freeze > requirements.txt). Add comments/docstrings explaining complex logic. Create a README.md explaining setup, how to run the pipeline and
Why: Makes the project easier for others (or future you) to understand, set up, and maintain.
Data Enrichment (if feasible):
Action: Investigate if appVersion and device information can be reliably extracted by the scraping libraries. If so, modify fetching and analysis scripts to include them, and ensure the dashboard filters work.
Why: Allows for deeper analysis to correlate feedback with specific app versions or device types.
Lower Priority (Advanced Features & UX Enhancements):
Deeper Analytics: Explore Sentiment Intensity analysis again, Competitor Comparison analysis (complex), or integrating Word Cloud stopwords.
Interactive Charts: Replace static charts with Plotly or Altair for enhanced exploration capabilities (hover, zoom).
Dashboard UX: Add saved filter views, better in-app help text/documentation.
This provides a clear path from the current functional dashboard to a more professional, automated, and robust internal analytics tool.
