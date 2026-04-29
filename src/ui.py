import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# Configure page
LOGO_PATH = Path(__file__).parent / "patrkaar_ai_logo.jpeg"
if not LOGO_PATH.exists():
    LOGO_PATH = Path(__file__).parent.parent / "patrkaar_ai_logo.jpeg"



# Header with logo
if LOGO_PATH.exists():
    try:
        st.image(str(LOGO_PATH), width=80)
    except Exception:
        pass

st.title("Patrakaar.AI News Intelligence Dashboard")

# Custom CSS for confidence color-coding
st.markdown("""
<style>
.confidence-high { color: #00ff00; font-weight: bold; }
.confidence-medium { color: #ffff00; font-weight: bold; }
.confidence-low { color: #ff0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Load data
DATA_PATH = Path("output/articles_output.json")
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error("Output file not found. Run processor first.")
        return None
    df = pd.read_json(DATA_PATH)
    # Convert tags to list if pipe-separated
    if df["tags"].dtype == object:
        df["tags"] = df["tags"].apply(lambda x: x.split("|") if isinstance(x, str) else x)
    return df

df = load_data()
if df is None:
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
topics = ["All"] + sorted(df["topic"].unique().tolist())
selected_topic = st.sidebar.selectbox("Filter by Topic", topics)

# Confidence filter
confidence_threshold = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.05)

# Apply filters
filtered_df = df.copy()
if selected_topic != "All":
    filtered_df = filtered_df[filtered_df["topic"] == selected_topic]
filtered_df = filtered_df[filtered_df["confidence"] >= confidence_threshold]

# Main content
st.metric("Total Articles", len(filtered_df))

# Analytics panel
st.subheader("Analytics")
col1, col2 = st.columns(2)

with col1:
    st.write("**Topic Distribution**")
    topic_counts = filtered_df["topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    fig = px.bar(topic_counts, x="Topic", y="Count", color="Topic", title="Articles per Topic")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write("**Confidence Distribution**")
    st.metric("Average Confidence", f"{filtered_df['confidence'].mean():.4f}")
    st.metric("Low Confidence Articles", int(filtered_df["low_confidence"].sum()))

# Articles table
st.subheader("Articles")
for _, row in filtered_df.iterrows():
    conf = row["confidence"]
    # Confidence color mapping
    if conf >= 0.8:
        conf_color = "#00ff00"
    elif conf >= 0.6:
        conf_color = "#ffff00"
    else:
        conf_color = "#ff0000"

    with st.expander(f"**{row['title']}** | Topic: {row['topic']} | Confidence: {conf:.4f}"):
        st.markdown(f"<span style='color: {conf_color}; font-weight: bold;'>Confidence: {conf:.4f}</span>", unsafe_allow_html=True)
        st.write(f"**Summary:** {row['summary']}")
        st.write(f"**Tags:** {', '.join(row['tags'])}")
        st.write(f"**Original Length:** {row['original_length']} chars | **Summary Length:** {row['summary_length']} chars")
        st.write(f"**Compression Ratio:** {row['compression_ratio']:.4f}")
        if row["low_confidence"]:
            st.warning("Low confidence classification")
