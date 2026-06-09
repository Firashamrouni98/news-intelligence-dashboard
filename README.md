# 📰 AI-Powered News Intelligence Dashboard

A real-time news intelligence dashboard powered by HuggingFace Transformers and NewsAPI.
Analyzes media sentiment across any topic using state-of-the-art NLP models, delivering actionable insights through an interactive visualization interface.
Built as a portfolio project demonstrating end-to-end AI engineering skills.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo
> Coming soon — deploying on Streamlit Cloud

---

## 📌 Features

- 🔍 Search any topic and fetch latest news articles via NewsAPI
- 🤖 Sentiment analysis using `cardiffnlp/twitter-roberta-base-sentiment-latest` — a transformer model fine-tuned on real-world social and news content
- 📊 Interactive dashboard with 4 visualizations:
  - Sentiment distribution (pie chart)
  - Confidence score distribution (histogram)
  - Top news sources (bar chart)
  - Articles over time (timeline)
- 💡 Auto-generated insight summary based on analysis results
- 🏆 Top positive and negative articles with direct links
- ⬇️ Export results as CSV

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| NLP Model | cardiffnlp/twitter-roberta-base-sentiment-latest |
| News Data | NewsAPI |
| Data Processing | Pandas |
| Visualization | Plotly |
| Language | Python 3.11 |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/news-intelligence-dashboard.git
cd news-intelligence-dashboard
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a free NewsAPI key
Sign up at [newsapi.org](https://newsapi.org) — free tier gives 100 requests/day.

### 5. Run the app
```bash
streamlit run app.py
```

Enter your NewsAPI key in the sidebar, choose a topic, and click **Analyze Now**.

---

## 📁 Project Structure
news-intelligence-dashboard/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation


---

## 🧠 How It Works
User enters topic
↓
NewsAPI fetches latest articles
↓
Title + Description sent to RoBERTa model
↓
Sentiment classified: Positive / Neutral / Negative
↓
Results stored in DataFrame
↓
Dashboard renders charts + insight summary

---

## 📊 Example Output

**Topic:** `artificial intelligence`

> 📊 Out of 15 articles analyzed about 'artificial intelligence',
> 60% were positive and 27% were neutral, suggesting optimistic
> media coverage around this topic. Average model confidence: 87.3%.
> Most active source: TechCrunch.

---

## 🔮 Future Improvements

- [ ] Add keyword extraction per article
- [ ] Track sentiment trends over time (database storage)
- [ ] Multi-language support
- [ ] Email digest of daily sentiment report
- [ ] Compare sentiment across multiple topics side by side

---

## 👤 Author

**Firas Hamrouni**
Telecommunications Engineering Graduate | AI & ML Engineer

[![LinkedIn](https://www.linkedin.com/in/firas-hamrouni-76177a277)]

---

## 📄 License

MIT License — free to use and modify.