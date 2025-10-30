# app.py
"""
Daily Nutrition & Mood Analytics Dashboard (Streamlit)
- No paid APIs, uses a local mock nutrition database.
- Mood detection uses VADER (no external downloads).
- Stores entries locally in a CSV file (entries.csv).
- Visualizations are interactive with Plotly.
- Run with: streamlit run app.py
"""

from datetime import datetime, date, timedelta
import json
import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------
# Configuration / constants
# -------------------------
DATA_FILE = "entries.csv"   # local storage
DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Simple mock nutrition database (per typical serving)
# Values: calories, protein (g), carbs (g), fat (g), fiber (g), sugar (g)
MOCK_NUTRITION_DB = {
    "egg": {"calories": 78, "protein": 6.3, "carbs": 0.6, "fat": 5.3, "fiber": 0.0, "sugar": 0.0},
    "eggs": {"calories": 78, "protein": 6.3, "carbs": 0.6, "fat": 5.3, "fiber": 0.0, "sugar": 0.0},
    "bread": {"calories": 80, "protein": 3.0, "carbs": 14.0, "fat": 1.0, "fiber": 1.0, "sugar": 1.5},
    "slice of bread": {"calories": 80, "protein": 3.0, "carbs": 14.0, "fat": 1.0, "fiber": 1.0, "sugar": 1.5},
    "milk": {"calories": 122, "protein": 8.0, "carbs": 12.0, "fat": 5.0, "fiber": 0.0, "sugar": 12.0},
    "banana": {"calories": 105, "protein": 1.3, "carbs": 27.0, "fat": 0.3, "fiber": 3.1, "sugar": 14.4},
    "apple": {"calories": 95, "protein": 0.5, "carbs": 25.0, "fat": 0.3, "fiber": 4.4, "sugar": 19.0},
    "oatmeal": {"calories": 150, "protein": 5.0, "carbs": 27.0, "fat": 3.0, "fiber": 4.0, "sugar": 1.0},
    "rice": {"calories": 206, "protein": 4.3, "carbs": 45.0, "fat": 0.4, "fiber": 0.6, "sugar": 0.1},
    "chicken breast": {"calories": 165, "protein": 31.0, "carbs": 0.0, "fat": 3.6, "fiber": 0.0, "sugar": 0.0},
    "salad": {"calories": 33, "protein": 2.0, "carbs": 6.0, "fat": 0.5, "fiber": 2.5, "sugar": 2.5},
    "avocado": {"calories": 234, "protein": 2.9, "carbs": 12.5, "fat": 21.0, "fiber": 10.0, "sugar": 1.0},
    "cheese": {"calories": 113, "protein": 7.0, "carbs": 0.9, "fat": 9.4, "fiber": 0.0, "sugar": 0.1},
    "butter": {"calories": 102, "protein": 0.1, "carbs": 0.0, "fat": 11.5, "fiber": 0.0, "sugar": 0.0},
    "yogurt": {"calories": 59, "protein": 10.0, "carbs": 3.6, "fat": 0.4, "fiber": 0.0, "sugar": 3.2},
    "orange": {"calories": 62, "protein": 1.2, "carbs": 15.4, "fat": 0.2, "fiber": 3.1, "sugar": 12.2},
    "coffee": {"calories": 2, "protein": 0.3, "carbs": 0.0, "fat": 0.0, "fiber": 0.0, "sugar": 0.0},
    "tea": {"calories": 2, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0, "sugar": 0.0},
    "sandwich": {"calories": 300, "protein": 12.0, "carbs": 30.0, "fat": 12.0, "fiber": 2.0, "sugar": 4.0},
    "pizza": {"calories": 285, "protein": 12.0, "carbs": 36.0, "fat": 10.0, "fiber": 2.5, "sugar": 4.0},
    "burger": {"calories": 354, "protein": 17.0, "carbs": 29.0, "fat": 20.0, "fiber": 1.5, "sugar": 6.0},
    "fries": {"calories": 312, "protein": 3.4, "carbs": 41.0, "fat": 15.0, "fiber": 3.8, "sugar": 0.3},
}

# If unknown food item, fallback nutrition (per serving)
FALLBACK_NUTRITION = {"calories": 120, "protein": 3.0, "carbs": 15.0, "fat": 5.0, "fiber": 1.0, "sugar": 2.0}

# Sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# -------------------------
# Helper functions
# -------------------------


def ensure_data_file():
    """Create the CSV with sample data if missing. This helps first run."""
    if not os.path.exists(DATA_FILE):
        sample_entries = []
        now = datetime.now()
        sample_entries.append(make_entry(
            "2 eggs, a slice of bread, a glass of milk",
            "Feeling good and energized",
            timestamp=(now - timedelta(days=1, hours=3))
        ))
        sample_entries.append(make_entry(
            "Chicken breast with rice and salad",
            "A bit tired but okay",
            timestamp=(now - timedelta(days=1))
        ))
        sample_entries.append(make_entry(
            "Banana and yogurt",
            "happy and light",
            timestamp=now - timedelta(hours=2)
        ))
        df = pd.DataFrame(sample_entries)
        df.to_csv(DATA_FILE, index=False)


def save_entry(entry):
    """Append a single entry (dict) to the CSV storage."""
    df = pd.DataFrame([entry])
    if os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(DATA_FILE, index=False)


def load_entries() -> pd.DataFrame:
    """Load entries CSV into DataFrame. Convert JSON columns back to dicts."""
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=[
            "timestamp", "food_text", "mood_text", "mood_label", "nutrition_total", "items"
        ])
    df = pd.read_csv(DATA_FILE)
    # Convert json-like string columns back to objects
    if "nutrition_total" in df.columns:
        df["nutrition_total"] = df["nutrition_total"].apply(lambda x: json.loads(x) if pd.notna(x) else {})
    if "items" in df.columns:
        df["items"] = df["items"].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def parse_food_items(food_text: str):
    """
    Very simple parser: split description by commas and ' and '.
    Try to detect quantity (e.g., '2 eggs') and match food keywords in the DB.
    Returns list of (servings, item_name, raw_text)
    """
    # Split on commas and ' and ' (case-insensitive)
    parts = re.split(r",|\band\b", food_text, flags=re.IGNORECASE)
    items = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Try to detect a leading number (quantity)
        qty = 1.0
        m = re.match(r"^(\d+(\.\d+)?)(\s+|x)?(.+)$", part)
        if m:
            qty = float(m.group(1))
            item_name = m.group(4).strip()
        else:
            # try "a", "an", "one"
            m2 = re.match(r"^(a|an|one)\s+(.+)$", part, flags=re.IGNORECASE)
            if m2:
                qty = 1.0
                item_name = m2.group(2).strip()
            else:
                item_name = part

        items.append((qty, item_name, part))
    return items


def match_nutrition_for_item(item_text: str):
    """
    Find the best match in MOCK_NUTRITION_DB by keyword presence.
    Lowercased substring matching; otherwise fallback.
    """
    low = item_text.lower()
    # attempt exact phrase matches first
    for key in MOCK_NUTRITION_DB.keys():
        if key in low:
            return MOCK_NUTRITION_DB[key].copy(), key
    # attempt token matching
    tokens = re.findall(r"[a-zA-Z]+", low)
    for t in tokens:
        if t in MOCK_NUTRITION_DB:
            return MOCK_NUTRITION_DB[t].copy(), t
    # fallback
    return FALLBACK_NUTRITION.copy(), None


def estimate_nutrition(food_text: str):
    """
    Estimate nutrition for the whole food_text.
    Returns:
      - items: list of dicts {name, qty, nutrition}
      - total: dict sum of nutrition
    """
    parsed = parse_food_items(food_text)
    items_out = []
    total = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, "sugar": 0}
    for qty, item_name, raw in parsed:
        nut, matched_key = match_nutrition_for_item(item_name)
        # multiply by qty (assume qty represents servings)
        item_nut = {k: round(v * qty, 2) for k, v in nut.items()}
        items_out.append({
            "raw": raw,
            "name": item_name,
            "matched": matched_key,
            "qty": qty,
            "nutrition": item_nut
        })
        for k in total:
            total[k] += item_nut.get(k, 0)
    # Round totals
    total = {k: round(v, 2) for k, v in total.items()}
    return items_out, total


def detect_mood(mood_text: str):
    """
    Use VADER to extract sentiment compound score and map to simple mood labels.
    Returns: label string and score
    """
    if not mood_text or str(mood_text).strip() == "":
        return "neutral", 0.0
    vs = analyzer.polarity_scores(mood_text)
    c = vs["compound"]
    # mapping
    if c >= 0.55:
        label = "very positive"
    elif c >= 0.15:
        label = "positive"
    elif c > -0.15:
        label = "neutral"
    elif c > -0.55:
        label = "negative"
    else:
        label = "very negative"
    return label, round(c, 3)


def make_entry(food_text, mood_text, timestamp=None):
    """Construct an entry dict ready to save."""
    if timestamp is None:
        timestamp = datetime.now()
    items, total = estimate_nutrition(food_text)
    mood_label, mood_score = detect_mood(mood_text)
    entry = {
        "timestamp": timestamp.strftime(DATE_FMT),
        "food_text": food_text,
        "mood_text": mood_text,
        "mood_label": mood_label,
        # store JSON strings so CSV stays tidy
        "nutrition_total": json.dumps(total),
        "items": json.dumps(items),
        "mood_score": mood_score
    }
    return entry


# -------------------------
# Streamlit App UI
# -------------------------
st.set_page_config(page_title="Nutrition & Mood Dashboard", page_icon="🍎", layout="wide")

# Sidebar with theming and navigation
st.sidebar.image(
    "https://raw.githubusercontent.com/streamlit/tutorials/main/images/food.png",
    use_column_width=True
) if st.sidebar.button("Show Example Image (optional)") else None

st.sidebar.title("Daily Nutrition & Mood")
st.sidebar.markdown("Track what you eat, analyze nutrition, and monitor mood trends.")

# Simple nav (tabs in main area)
tab = st.sidebar.radio("Go to", ["Add Entry ➕", "Analytics 📊", "History 🗂️"])

# Ensure storage exists and has sample data
ensure_data_file()
df_all = load_entries()

# -------------------------
# Tab: Add Entry
# -------------------------
if tab.startswith("Add"):
    st.title("Add Entry ➕")
    st.markdown("Log a meal and how you're feeling. The app will estimate nutrition and analyze mood automatically.")
    with st.form("entry_form"):
        food_input = st.text_area("What did you eat? (e.g., '2 eggs, a slice of bread, and a glass of milk')",
                                  placeholder="Write your meal description here...", height=120)
        mood_input = st.text_input("How are you feeling? (optional)", placeholder="e.g., feeling relaxed, tired...")
        submitted = st.form_submit_button("Save Entry")
    if submitted:
        if not food_input or not str(food_input).strip():
            st.error("Please enter what you ate.")
        else:
            entry = make_entry(food_input.strip(), mood_input.strip())
            save_entry(entry)
            st.success("Saved entry ✅")
            st.write("**Estimated nutrition (total):**")
            st.json(json.loads(entry["nutrition_total"]))
            st.write("**Detected mood:**", entry["mood_label"], "(score: {})".format(entry["mood_score"]))
            st.write("**Parsed items:**")
            st.json(json.loads(entry["items"]))

# -------------------------
# Tab: Analytics
# -------------------------
elif tab.startswith("Analytics"):
    st.title("Analytics Dashboard 📊")
    st.markdown("Visualize calories, macros, and mood trends over time.")

    df = load_entries()
    if df.empty:
        st.info("No entries yet. Add entries under 'Add Entry' tab.")
    else:
        # Expand items
        # Create a DataFrame of totals per entry
        totals = df[["timestamp", "food_text", "mood_label", "mood_score"]].copy()
        totals["nutrition_total"] = df["nutrition_total"]
        # Expand nutrition_total dicts into columns
        nut_df = totals["nutrition_total"].apply(pd.Series)
        totals = pd.concat([totals.drop(columns=["nutrition_total"]), nut_df], axis=1)
        totals["date"] = totals["timestamp"].dt.date
        # Today's totals
        today = date.today()
        today_mask = totals["date"] == today
        today_totals = totals[today_mask]
        # Summary metrics at top
        col1, col2, col3, col4 = st.columns(4)
        total_cal_today = int(today_totals["calories"].sum()) if not today_totals.empty else 0
        avg_protein = round(totals["protein"].mean(), 1) if not totals.empty else 0
        avg_carbs = round(totals["carbs"].mean(), 1) if not totals.empty else 0
        avg_fat = round(totals["fat"].mean(), 1) if not totals.empty else 0
        most_freq_mood = totals["mood_label"].mode().iloc[0] if not totals["mood_label"].empty else "N/A"

        col1.metric("Total calories today", f"{total_cal_today} kcal")
        col2.metric("Avg protein per entry", f"{avg_protein} g")
        col3.metric("Avg carbs per entry", f"{avg_carbs} g")
        col4.metric("Most frequent mood", most_freq_mood)

        st.markdown("---")

        # Daily calorie breakdown (bar chart) - calories per entry with timestamp
        st.subheader("Calorie breakdown (entries)")
        if not totals.empty:
            fig_bar = px.bar(
                totals.sort_values("timestamp"),
                x="timestamp",
                y="calories",
                hover_data=["food_text", "protein", "carbs", "fat"],
                labels={"timestamp": "Entry time", "calories": "Calories (kcal)"},
                title="Calories per entry"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Not enough data for calorie chart.")

        # Macro-nutrient distribution pie chart for selected date
        st.subheader("Macro-nutrient distribution")
        date_sel = st.date_input("Select date for macro distribution", value=today)
        selected = totals[totals["date"] == date_sel]
        if selected.empty:
            st.info("No entries for this date.")
        else:
            macros = {
                "Protein (g)": selected["protein"].sum(),
                "Carbs (g)": selected["carbs"].sum(),
                "Fat (g)": selected["fat"].sum()
            }
            fig_pie = px.pie(names=list(macros.keys()), values=list(macros.values()), title=f"Macros on {date_sel}")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        # Mood trend over time
        st.subheader("Mood trend over time")
        # We'll show counts of moods per day over last 30 days (or available range)
        last_n_days = st.slider("Days to show", min_value=7, max_value=90, value=30)
        start_date = date.today() - timedelta(days=last_n_days - 1)
        mood_df = totals.copy()
        mood_df["date_only"] = mood_df["date"]
        mood_df = mood_df[mood_df["date_only"] >= start_date]
        if mood_df.empty:
            st.info("Not enough mood data in the selected range.")
        else:
            mood_counts = mood_df.groupby(["date_only", "mood_label"]).size().reset_index(name="count")
            # Pivot to have mood labels as columns
            mood_pivot = mood_counts.pivot(index="date_only", columns="mood_label", values="count").fillna(0)
            fig = go.Figure()
            for mood_col in mood_pivot.columns:
                fig.add_trace(go.Scatter(x=mood_pivot.index, y=mood_pivot[mood_col],
                                         mode="lines+markers", name=mood_col))
            fig.update_layout(title="Mood counts per day", xaxis_title="Date", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        # Show raw totals table (last 50 entries)
        st.subheader("Recent entries (last 50)")
        display_df = totals.sort_values("timestamp", ascending=False).head(50)
        display_df_for_table = display_df[["timestamp", "food_text", "calories", "protein", "carbs", "fat", "mood_label"]]
        st.dataframe(display_df_for_table, use_container_width=True)

# -------------------------
# Tab: History
# -------------------------
elif tab.startswith("History"):
    st.title("History 🗂️")
    df = load_entries()
    if df.empty:
        st.info("No history found. Add entries first.")
    else:
        # Provide filtering and export
        df_display = df.copy()
        df_display["date"] = df_display["timestamp"].dt.date
        # Filter by date range
        min_date = df_display["date"].min()
        max_date = df_display["date"].max()
        start = st.date_input("Start date", value=min_date)
        end = st.date_input("End date", value=max_date)
        mask = (df_display["date"] >= start) & (df_display["date"] <= end)
        filtered = df_display[mask].sort_values("timestamp", ascending=False)

        st.write(f"Showing {len(filtered)} entries from {start} to {end}.")
        # Expand items for readability
        def stringify_items(items_json):
            try:
                items = items_json if isinstance(items_json, list) else json.loads(items_json)
                parts = []
                for it in items:
                    parts.append(f"{it.get('qty',1)}×{it.get('matched') or it.get('name')} ({it.get('nutrition')})")
                return " | ".join(parts)
            except Exception:
                return str(items_json)

        table = filtered[["timestamp", "food_text", "mood_text", "mood_label", "nutrition_total", "items"]].copy()
        table["nutrition_total"] = table["nutrition_total"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
        table["items_pretty"] = table["items"].apply(stringify_items)
        st.dataframe(table[["timestamp", "food_text", "mood_text", "mood_label", "items_pretty"]], use_container_width=True)

        # Allow export filtered CSV
        csv = filtered.to_csv(index=False)
        st.download_button("Download filtered entries (CSV)", data=csv, file_name="nutrition_mood_history.csv", mime="text/csv")

# -------------------------
# Footer / small tips
# -------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit. Nutrition estimates are approximate — use them as guidance, not medical advice.")
