import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
from tqdm import tqdm

# Constants
MIN_CONTENT_LENGTH = 50
MAX_INPUT_TOKENS = 1024
DEFAULT_TOP_KEYWORDS = 5
CANDIDATE_LABELS = ["Politics", "Business", "Technology", "Sports", "Entertainment"]
LOW_CONFIDENCE_THRESHOLD = 0.6
MODEL_SUMMARY = "sshleifer/distilbart-cnn-12-6"
MODEL_CLASSIFY = "facebook/bart-large-mnli"
KEYBERT_BACKEND = "all-MiniLM-L6-v2"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lazy model singletons
_summary_pipeline = None
_keybert_model = None
_classifier_pipeline = None


def get_summary_pipeline():
    """Load and return summarization pipeline singleton."""
    global _summary_pipeline
    if _summary_pipeline is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _summary_pipeline = pipeline("summarization", model=MODEL_SUMMARY, device=device)
        logger.info(f"Loaded summarization model on device {device}")
    return _summary_pipeline


def get_keybert_model():
    """Load and return KeyBERT model singleton."""
    global _keybert_model
    if _keybert_model is None:
        from keybert import KeyBERT
        _keybert_model = KeyBERT(model=KEYBERT_BACKEND)
        logger.info("Loaded KeyBERT model")
    return _keybert_model


def get_classifier_pipeline():
    """Load and return zero-shot classification pipeline singleton."""
    global _classifier_pipeline
    if _classifier_pipeline is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _classifier_pipeline = pipeline("zero-shot-classification", model=MODEL_CLASSIFY, device=device)
        logger.info(f"Loaded classification model on device {device}")
    return _classifier_pipeline


def load_articles(input_path: Path) -> pd.DataFrame:
    """Load and clean article data from Excel file."""
    logger.info(f"Loading articles from {input_path}")
    df = pd.read_excel(input_path)
    df.columns = df.columns.str.lower()  # Normalize to lowercase
    
    # Fallback: use title as content if content is missing/short
    mask = df["content"].isna() | (df["content"].str.len() < MIN_CONTENT_LENGTH)
    df.loc[mask, "content"] = df.loc[mask, "title"]
    logger.info(f"Loaded {len(df)} articles")
    return df


def summarize_article(content: str, max_input_tokens: int = MAX_INPUT_TOKENS) -> Dict:
    """Generate summary and compression ratio for article content."""
    original_len = len(content)
    if original_len < MIN_CONTENT_LENGTH:
        return {"summary": content, "compression_ratio": 1.0}
    
    pipeline = get_summary_pipeline()
    # Token-based truncation to avoid exceeding model max length
    tokenizer = pipeline.tokenizer
    tokens = tokenizer.encode(content, truncation=True, max_length=max_input_tokens)
    truncated = tokenizer.decode(tokens, skip_special_tokens=True)
    result = pipeline(truncated, max_length=150, min_length=50, do_sample=False)
    summary = result[0]["summary_text"]
    summary_len = len(summary)
    compression = round(summary_len / original_len, 4) if original_len > 0 else 1.0
    return {"summary": summary, "compression_ratio": compression}


def extract_keywords(content: str, top_n: int = DEFAULT_TOP_KEYWORDS) -> List[str]:
    """Extract semantic keywords using KeyBERT with MMR diversity."""
    keywords = get_keybert_model().extract_keywords(
        content,
        top_n=top_n,
        use_mmr=True,
        diversity=0.7
    )
    return [kw[0] for kw in keywords]


def classify_topic(content: str, candidate_labels: List[str] = CANDIDATE_LABELS) -> Dict:
    """Classify article topic using zero-shot BART-MNLI."""
    classifier = get_classifier_pipeline()
    tokenizer = classifier.tokenizer
    tokens = tokenizer.encode(content, truncation=True, max_length=512)
    truncated = tokenizer.decode(tokens, skip_special_tokens=True)
    result = classifier(truncated, candidate_labels=candidate_labels)
    return {
        "topic": result["labels"][0],
        "confidence": round(result["scores"][0], 4)
    }


def process_all_articles(df: pd.DataFrame) -> List[Dict]:
    """Process all articles through NLP pipeline and return structured results."""
    results = []
    logger.info("Starting article processing")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
        content = str(row["content"])
        summary_data = summarize_article(content)
        keywords = extract_keywords(content)
        topic_data = classify_topic(content)
        
        results.append({
            "title": str(row["title"]),
            "summary": summary_data["summary"],
            "tags": keywords,
            "topic": topic_data["topic"],
            "confidence": topic_data["confidence"],
            "low_confidence": topic_data["confidence"] < LOW_CONFIDENCE_THRESHOLD,
            "compression_ratio": summary_data["compression_ratio"],
            "original_length": len(content),
            "summary_length": len(summary_data["summary"])
        })
    logger.info(f"Processed {len(results)} articles")
    return results


def save_output(results: List[Dict], output_dir: Path) -> None:
    """Save results to JSON and CSV, with idempotency check."""
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "articles_output.json"
    csv_path = output_dir / "articles_output.csv"
    
    # Idempotency check
    if json_path.exists():
        existing = pd.read_json(json_path)
        if len(existing) == len(results):
            logger.info("Output already exists and matches input count, skipping save")
            return
    
    # Save JSON
    pd.DataFrame(results).to_json(json_path, orient="records", indent=2)
    # Save CSV (tags as pipe-separated)
    df = pd.DataFrame(results)
    df["tags"] = df["tags"].apply(lambda x: "|".join(x))
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved output to {json_path} and {csv_path}")


def main():
    """CLI entry point for processing articles."""
    parser = argparse.ArgumentParser(description="Process news articles NLP pipeline")
    parser.add_argument("--input", type=Path, default=Path("data/raw_data.xlsx"), help="Input Excel file path")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory path")
    args = parser.parse_args()
    
    df = load_articles(args.input)
    results = process_all_articles(df)
    save_output(results, args.output)


if __name__ == "__main__":
    main()
