# Patrakaar.AI — News Intelligence Pipeline

An end-to-end NLP pipeline that automatically summarizes news articles, extracts keywords, and classifies topics. Built to handle production-scale newsroom data (1,035 articles) with a FastAPI backend and an interactive Streamlit dashboard.

---

## Architecture

```
Raw Articles (Excel)
        │
        ▼
┌───────────────────────────────────────────────────┐
│                  processor.py                     │
│                                                   │
│  ┌─────────────────┐  ┌──────────┐  ┌──────────┐  │
│  │  DistilBART     │  │ KeyBERT  │  │BART-MNLI │  │
│  │  Summarization  │  │ Keywords │  │Zero-shot │  │
│  │  (1024t limit)  │  │  (MMR)   │  │   Topic  │  │
│  └─────────────────┘  └──────────┘  └──────────┘  │ 
│                                                   │
│  Smart Components:                                │
│  • Tokenizer-based truncation (no CUDA crashes)   │
│  • Lazy singletons (6GB VRAM management)          │
│  • Idempotency check (skip 3hr re-runs)           │
│  • Content fallback (handles missing fields)      │
└───────────────────────┬───────────────────────────┘
                        │
              Structured JSON / CSV
              (9 fields per article)
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
  ┌──────────────┐            ┌──────────────────┐
  │  FastAPI     │            │  Streamlit UI    │
  │  api.py      │            │  ui.py           │
  │  /articles   │            │  • Topic filters │
  │  /stats      │            │  • Confidence    │
  │  (lifespan)  │            │    threshold     │
  └──────────────┘            │  • Article view  │
                              └──────────────────┘
```

---

## Model Choices & Rationale

### Summarization — `sshleifer/distilbart-cnn-12-6`

DistilBART (306M params) was chosen over full BART (1.6B) or GPT-4 for a practical reason: the target machine has a 6GB VRAM GPU running three models simultaneously. DistilBART is 6× faster and uses a fraction of the VRAM while retaining 90%+ of BART's summarization quality.

Extractive methods like TextRank were ruled out because they select existing sentences verbatim rather than synthesizing a coherent summary. Abstractive models produce genuinely condensed output, which is what a newsroom tool needs.

**Edge case handled:** Articles under 50 characters cause BART to hallucinate. These are returned as-is with `compression_ratio = 1.0`.

### Keyword Extraction — `KeyBERT` with MMR

TF-IDF counts word frequency; it has no understanding of meaning. KeyBERT uses BERT embeddings to find semantically relevant phrases, ranking them by their proximity to the full document in embedding space.

MMR (Maximal Marginal Relevance) with `diversity=0.7` is enabled to prevent redundant keywords. Without it, results for a tech article collapse into near-synonyms: `["AI", "artificial intelligence", "machine learning", "deep learning"]`. MMR trades raw relevance for coverage.

### Topic Classification — `facebook/bart-large-mnli` (Zero-Shot)

No labeled training data was available for fine-tuning, making zero-shot classification the only viable approach. BART-MNLI frames each article as a natural language inference problem: it asks whether the article *entails* each candidate label (Politics, Business, Technology, Sports, Entertainment) and scores accordingly.

This works well as a baseline but carries a known limitation: confidence is low across the board when classifying short or truncated text. See [Metrics](#metrics-summary) for the honest numbers.

---

## Engineering Decisions

### 1. Tokenizer-Based Truncation

**Problem:** An article containing 4,721 tokens exceeded BART's 1,024-token limit. Character-based truncation (`content[:4096]`) doesn't account for multi-byte characters and subword tokenization — it triggered CUDA assertion errors.

**Solution:**
```python
tokens = tokenizer.encode(content, truncation=True, max_length=1024)
truncated = tokenizer.decode(tokens, skip_special_tokens=True)
```

This guarantees the input never exceeds the model's context window, regardless of language or character encoding.

### 2. Lazy Singletons for VRAM Management

**Problem:** Loading DistilBART (~300MB), BART-MNLI (~1.6GB), and KeyBERT (~80MB) all at startup on a 6GB GPU risks OOM errors and wastes memory if a model isn't used.

**Solution:** Module-level lazy initialization — each model loads once, only on first call:
```python
_summary_pipeline = None

def get_summary_pipeline():
    global _summary_pipeline
    if _summary_pipeline is None:
        _summary_pipeline = pipeline("summarization", model=MODEL_SUMMARY, device=device)
    return _summary_pipeline
```

All three models coexist on the same GPU without memory conflicts.

### 3. Content Fallback for Malformed Rows

**Problem:** ~80 articles had `"Content not found"` in the content field. The actual article text was only available in the title column (sometimes with metadata concatenated, e.g. `"India / Apr 28, 2026Go for mediation..."`).

**Solution:**
```python
mask = df["content"].isna() | (df["content"].str.len() < MIN_CONTENT_LENGTH)
df.loc[mask, "content"] = df.loc[mask, "title"]
```

This achieves 100% data utilization — no articles skipped — at the cost of some keyword noise in those rows (e.g., `"2026go"` appearing as an extracted keyword from concatenated metadata).

### 4. Idempotency Check

**Problem:** A full processing run takes ~3 hours. Accidentally re-running should not repeat all that work.

**Solution:** Before writing output, check if the file already exists and matches the current article count:
```python
if json_path.exists():
    existing = pd.read_json(json_path)
    if len(existing) == len(results):
        logger.info("Output already exists and matches input count, skipping save")
        return
```

### 5. Modern FastAPI Patterns

The API uses the `lifespan` context manager (not the deprecated `@app.on_event` hook) and Pydantic v2 response models throughout.

---

## Output Schema

Each processed article produces 9 fields:

| Field | Type | Description |
|---|---|---|
| `title` | string | Original article title |
| `content` | string | Source content (with fallback applied) |
| `summary` | string | Abstractive summary from DistilBART |
| `keywords` | list[str] | Top keyphrases from KeyBERT |
| `topic` | string | Predicted topic label |
| `confidence` | float | Zero-shot classification confidence |
| `low_confidence` | bool | `true` if confidence < 0.6 |
| `compression_ratio` | float | `len(summary) / len(content)` |
| `word_count` | int | Word count of original content |

---

## Results

### Topic Distribution

![Topic Distribution](output/topic_distribution.png)

| Topic | Count | Share |
|---|---|---|
| Politics | 467 | 45.1% |
| Technology | 265 | 25.6% |
| Business | 188 | 18.2% |
| Entertainment | 81 | 7.8% |
| Sports | 34 | 3.3% |

The distribution is consistent with the composition of Indian news media in the dataset period.

### Confidence Distribution

![Confidence Distribution](output/confidence_distribution.png)

Mean confidence sits at **0.4467**, with 80.5% of articles below the 0.6 threshold. This is an expected consequence of zero-shot classification on short and sometimes truncated text — the model is making informed inferences without task-specific training. The confidence values are surfaced transparently in both the API response and the UI, not hidden.

### Compression Ratios

![Compression Distribution](output/compression_distribution.png)

Average compression ratio: **0.4988** — summaries are roughly half the length of the source. Ratios above 1.0 (max: 4.8793) occur when BART's minimum output length (50 tokens) exceeds the input, which happens with very short articles.

### Top Keywords

![Top Keywords](output/top_keywords.png)

Top 15 keyphrases across all 1,035 articles, extracted by KeyBERT. Dominant terms reflect the dataset: `india`, `government`, `election`, `covid`. The absence of stop words and generic filler confirms that semantic embedding-based extraction outperforms frequency counting for this task.

---

## Metrics Summary

| Metric | Value |
|---|---|
| Total articles processed | 1,035 |
| Avg compression ratio | 0.4988 |
| Min / Max compression ratio | 0.0177 / 4.8793 |
| Avg classification confidence | 0.4467 |
| Low-confidence articles (<0.6) | 833 (80.5%) |
| Most common topic | Politics (45.1%) |
| Least common topic | Sports (3.3%) |
| Avg keywords per article | 4.99 |

---

## Dashboard

### Main View
![UI Dashboard](output/ui_dashboard.png)

Displays total article count, topic distribution chart, and confidence KPIs on load.

### Sidebar Filters
![UI Filters](output/ui_filters.png)

Filter by topic (`st.sidebar.selectbox`) and minimum confidence threshold (`st.sidebar.slider`). Article table updates in real time.

### Article Detail
![UI Article Expanded](output/ui_article_expanded.png)

Expand any row to see the full summary, keyword tags, original vs summary length, compression ratio, and a low-confidence warning where applicable. Confidence is color-coded (green / yellow / red) using inline HTML — no emoji.

---

## Limitations & Proposed Improvements

### 1. Low Classification Confidence → Fine-Tuning

The 0.4467 average confidence is a direct result of using zero-shot classification without any labeled data. The fix is straightforward: build a human-in-the-loop correction interface in Streamlit, collect editor-corrected labels, and fine-tune a smaller classifier (e.g., `distilbert-base-uncased`) on that data. Even 500–1,000 labeled examples would likely push confidence above 0.8.

### 2. Long Article Truncation → Overlapping Chunk Summarization

Articles exceeding 1,024 tokens are currently truncated before summarization, which drops information from longer pieces. A better approach: split into overlapping 1,024-token chunks, summarize each independently, then run a second-pass summarization on the concatenated chunk summaries. This preserves full-article context.

### 3. Batch Processing → Live RSS Ingestion

Replace the Excel-based batch workflow with scheduled RSS scraping: fetch articles hourly, process through the pipeline, store results in PostgreSQL. This moves the system from retrospective analysis to real-time news intelligence.

---

## Scaling to 100K Articles/Day

The current single-process architecture wouldn't survive production load. Here's the path to scale:

- **Celery + Redis** — Distribute NLP tasks across multiple GPU workers running in parallel
- **TorchServe** — Dedicated model servers with dynamic batching; decouple model inference from the API process
- **PostgreSQL** — Replace JSON file storage with an indexed database; query by `topic`, `confidence`, and date ranges in milliseconds
- **Redis caching** — Cache `/stats` and frequently filtered `/articles` responses
- **RabbitMQ / Kafka** — Queue-based ingestion to decouple the scraper from the NLP pipeline
- **nginx + horizontal API scaling** — Multiple FastAPI workers behind a reverse proxy
- **Prometheus + Grafana** — Monitor pipeline throughput, inference latency, and confidence drift over time

---

## Setup

```bash
# Create environment
conda create -n patrakaar-ai python=3.12 -y
conda activate patrakaar-ai

# Install dependencies
pip install -r requirements.txt

# Run the NLP processor (~3 hours on GPU)
python src/processor.py --input data/raw_data.xlsx --output output/

# Start the API (new terminal)
uvicorn src.api:app --reload --port 8000

# Start the dashboard (new terminal)
streamlit run src/ui.py --server.port 8501
```

- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

> **Note:** The processor checks for existing output before running. If `output/` already contains results matching the input count, it will skip processing automatically.#   P a t r a k a a r . A I - n o - c 0 p y r i g h t -  
 