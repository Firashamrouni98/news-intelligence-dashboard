import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
from datetime import datetime

st.set_page_config(page_title="News Sentiment Analyzer", page_icon="📰", layout="wide")

st.markdown("""
<style>
    .insight-box {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-left: 4px solid #00d4ff;
        padding: 16px 20px;
        border-radius: 8px;
        color: white;
        font-size: 16px;
        line-height: 1.6;
    }
    .article-positive {
        border-left: 4px solid #2ecc71;
        padding: 10px 15px;
        margin: 6px 0;
        border-radius: 4px;
        background: rgba(46, 204, 113, 0.05);
    }
    .article-negative {
        border-left: 4px solid #e74c3c;
        padding: 10px 15px;
        margin: 6px 0;
        border-radius: 4px;
        background: rgba(231, 76, 60, 0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("📰 AI News Sentiment Analyzer")
st.caption("Real-time media sentiment analysis powered by HuggingFace Transformers + NewsAPI")

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("NewsAPI Key", type="password", placeholder="Paste your key here")
topic = st.sidebar.text_input("🔍 Search Topic", value="artificial intelligence")
num_articles = st.sidebar.slider("Number of Articles", min_value=5, max_value=30, value=15)
run = st.sidebar.button("🚀 Analyze Now", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("1. Fetches news via NewsAPI\n2. Analyzes sentiment with DistilBERT\n3. Visualizes results in real-time")
st.sidebar.markdown("[Get free NewsAPI key →](https://newsapi.org)")

if not run:
    st.info("👈 Enter your NewsAPI key and a topic, then click **Analyze Now** to start.")
    st.stop()

if not api_key:
    st.error("⚠️ Please enter your NewsAPI key in the sidebar.")
    st.stop()

# ── LOAD MODEL ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
       model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=1
    )

# ── FETCH NEWS ────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_news(query, page_size, key):
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "apiKey": key, "pageSize": page_size, "sortBy": "publishedAt", "language": "en"},
            timeout=10
        )
        data = r.json()
        if data.get("status") != "ok":
            return None, data.get("message", "API error")
        return data.get("articles", []), None
    except Exception as e:
        return None, str(e)

# ── ANALYZE ───────────────────────────────────────────────
def analyze_articles(articles, model):
    results = []
    for article in articles:
        title = article.get("title") or ""
        description = article.get("description") or ""
        if not title or title == "[Removed]":
            continue
        text = f"{title}. {description}"[:512]
        sentiment = model(text)[0][0]
        results.append({
            "Title": title,
            "Description": (description[:150] + "...") if len(description) > 150 else description,
            "Source": article.get("source", {}).get("name", "Unknown"),
            "Published": article.get("publishedAt", "")[:10],
            "Sentiment": sentiment["label"],
            "Confidence": round(sentiment["score"] * 100, 1),
            "URL": article.get("url", "#")
        })
    return pd.DataFrame(results)

# ── INSIGHT ───────────────────────────────────────────────
def generate_insight(df, topic):
    total = len(df)
    positive = len(df[df["Sentiment"] == "positive"])
    negative = len(df[df["Sentiment"] == "negative"])
    pos_pct = round(positive / total * 100)
    neg_pct = round(negative / total * 100)
    avg_conf = round(df["Confidence"].mean(), 1)
    top_source = df["Source"].value_counts().index[0]
    dominant = df["Sentiment"].value_counts().index[0]
    tone = {"positive": "optimistic", "negative": "concerning", "neutral": "mixed"}.get(dominant, "mixed")   
    return (
        f"📊 Out of **{total} articles** analyzed about **'{topic}'**, "
        f"**{pos_pct}% were positive** and **{neg_pct}% were negative**, "
        f"suggesting **{tone} media coverage** around this topic. "
        f"Average model confidence: **{avg_conf}%**. "
        f"Most active source: **{top_source}**."
    )

# ── RUN ───────────────────────────────────────────────────
with st.spinner("Loading AI model... (first run downloads ~250MB, be patient)"):
    model = load_model()

with st.spinner(f"Fetching articles about '{topic}'..."):
    articles, error = fetch_news(topic, num_articles, api_key)

if error:
    st.error(f"❌ NewsAPI error: {error}")
    st.stop()

if not articles:
    st.warning("No articles found. Try a different topic.")
    st.stop()

with st.spinner("Analyzing sentiment..."):
    df = analyze_articles(articles, model)

if df.empty:
    st.warning("All articles were filtered out. Try a different topic.")
    st.stop()

# ── METRICS ───────────────────────────────────────────────
st.markdown("### 📊 Overview")
c1, c2, c3, c4, c5 = st.columns(5)
positive_count = len(df[df["Sentiment"] == "positive"])
negative_count = len(df[df["Sentiment"] == "negative"])
neutral_count = len(df[df["Sentiment"] == "neutral"])
c1.metric("📰 Total Articles", len(df))
c2.metric("🟢 Positive", positive_count)
c3.metric("🔴 Negative", negative_count)
c4.metric("🟡 Neutral", neutral_count)
c5.metric("🎯 Avg Confidence", f"{round(df['Confidence'].mean(), 1)}%")

st.markdown("---")

# ── INSIGHT ───────────────────────────────────────────────
st.markdown("### 💡 Insight Summary")
st.markdown(f'<div class="insight-box">{generate_insight(df, topic)}</div>', unsafe_allow_html=True)

st.markdown("---")

# ── CHARTS ────────────────────────────────────────────────
st.markdown("### 📈 Visual Analysis")
col1, col2 = st.columns(2)

with col1:
    fig_pie = px.pie(
        df, names="Sentiment", title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#f39c12"},
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_hist = px.histogram(
        df, x="Confidence", color="Sentiment",
        title="Confidence Score Distribution",
        color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#f39c12"},
        nbins=20, barmode="overlay", opacity=0.75
    )
    st.plotly_chart(fig_hist, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    source_counts = df["Source"].value_counts().head(8).reset_index()
    source_counts.columns = ["Source", "Count"]
    fig_sources = px.bar(
        source_counts, x="Count", y="Source", orientation="h",
        title="Top News Sources", color="Count",
        color_continuous_scale="Blues"
    )
    fig_sources.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
    st.plotly_chart(fig_sources, use_container_width=True)

with col4:
    timeline_df = df.groupby(["Published", "Sentiment"]).size().reset_index(name="Count")
    fig_timeline = px.bar(
        timeline_df, x="Published", y="Count", color="Sentiment",
        title="Articles Over Time",
        color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#f39c12"},
        barmode="stack"
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

st.markdown("---")

# ── TOP ARTICLES ──────────────────────────────────────────
st.markdown("### 🏆 Top Articles")
tab1, tab2 = st.tabs(["🟢 Most Positive", "🔴 Most Negative"])

with tab1:
    top_pos = df[df["Sentiment"] == "positive"].nlargest(5, "Confidence")
    if top_pos.empty:
        st.info("No positive articles found.")
    for _, row in top_pos.iterrows():
        st.markdown(f"""
        <div class="article-positive">
            <strong><a href="{row['URL']}" target="_blank">{row['Title']}</a></strong><br>
            <small>📰 {row['Source']} &nbsp;|&nbsp; 📅 {row['Published']} &nbsp;|&nbsp; 🎯 {row['Confidence']}% confidence</small><br>
            <small>{row['Description']}</small>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    top_neg = df[df["Sentiment"] == "negative"].nlargest(5, "Confidence")
    if top_neg.empty:
        st.info("No negative articles found.")
    for _, row in top_neg.iterrows():
        st.markdown(f"""
        <div class="article-negative">
            <strong><a href="{row['URL']}" target="_blank">{row['Title']}</a></strong><br>
            <small>📰 {row['Source']} &nbsp;|&nbsp; 📅 {row['Published']} &nbsp;|&nbsp; 🎯 {row['Confidence']}% confidence</small><br>
            <small>{row['Description']}</small>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── RAW DATA ──────────────────────────────────────────────
with st.expander("🗂️ View Full Dataset"):
    st.dataframe(
        df[["Title", "Source", "Published", "Sentiment", "Confidence"]],
        use_container_width=True,
        hide_index=True
    )
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "sentiment_results.csv", "text/csv")