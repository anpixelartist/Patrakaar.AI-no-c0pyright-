from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional
import difflib
import pandas as pd

# Pydantic response models
class ArticleResponse(BaseModel):
    title: str
    summary: str
    tags: List[str]
    topic: str
    confidence: float
    low_confidence: bool
    compression_ratio: float
    original_length: int
    summary_length: int

class ArticlesListResponse(BaseModel):
    count: int
    articles: List[ArticleResponse]

class StatsResponse(BaseModel):
    total_articles: int
    topic_distribution: dict
    avg_confidence: float
    low_confidence_count: int
    avg_compression_ratio: float

# Global data store
articles_df = None
ARTICLES_PATH = Path("output/articles_output.json")
VALID_TOPICS = ["Politics", "Business", "Technology", "Sports", "Entertainment"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load article data on startup."""
    global articles_df
    if ARTICLES_PATH.exists():
        articles_df = pd.read_json(ARTICLES_PATH)
        # Convert tags from pipe-separated string to list if needed
        if articles_df["tags"].dtype == object:
            articles_df["tags"] = articles_df["tags"].apply(lambda x: x.split("|") if isinstance(x, str) else x)
        print(f"Loaded {len(articles_df)} articles from {ARTICLES_PATH}")
    else:
        print("Warning: Output file not found. Run processor first.")
    yield
    articles_df = None

app = FastAPI(title="Patrakaar.AI News API", lifespan=lifespan)

@app.get("/articles", response_model=ArticlesListResponse)
def get_articles(topic: Optional[str] = Query(None, description="Filter by topic (case-insensitive)")):
    """Return all articles, optionally filtered by topic."""
    if articles_df is None:
        raise HTTPException(status_code=503, detail="Article data not loaded")
    
    df = articles_df.copy()
    if topic:
        topic_lower = topic.lower()
        valid_topics_lower = [t.lower() for t in VALID_TOPICS]
        if topic_lower not in valid_topics_lower:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid topic. Valid topics: {VALID_TOPICS}"
            )
        df = df[df["topic"].str.lower() == topic_lower]
    
    articles = df.to_dict(orient="records")
    return {"count": len(articles), "articles": articles}

@app.get("/articles/{title}", response_model=ArticleResponse)
def get_article_by_title(title: str):
    """Return single article by title (fuzzy match)."""
    if articles_df is None:
        raise HTTPException(status_code=503, detail="Article data not loaded")
    
    # Exact match first
    exact = articles_df[articles_df["title"] == title]
    if not exact.empty:
        return exact.iloc[0].to_dict()
    
    # Fuzzy match
    titles = articles_df["title"].tolist()
    matches = difflib.get_close_matches(title, titles, n=1, cutoff=0.6)
    if matches:
        article = articles_df[articles_df["title"] == matches[0]].iloc[0]
        return article.to_dict()
    
    raise HTTPException(status_code=404, detail="Article not found")

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Return aggregate analytics for all articles."""
    if articles_df is None:
        raise HTTPException(status_code=503, detail="Article data not loaded")
    
    topic_dist = articles_df["topic"].value_counts().to_dict()
    avg_confidence = round(articles_df["confidence"].mean(), 4)
    low_conf_count = int(articles_df["low_confidence"].sum())
    avg_compression = round(articles_df["compression_ratio"].mean(), 4)
    
    return {
        "total_articles": len(articles_df),
        "topic_distribution": topic_dist,
        "avg_confidence": avg_confidence,
        "low_confidence_count": low_conf_count,
        "avg_compression_ratio": avg_compression
    }
