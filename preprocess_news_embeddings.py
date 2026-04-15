"""
preprocess_news_embeddings.py
─────────────────────────────
Pipeline: wheat_news.csv  →  daily FinBERT [CLS] embeddings (768-d)
                           →  PCA reduction (768-d → 16-d)
                           +  daily sentiment scores (positive / negative / neutral)

Steps
  1. Load & clean text: concatenate title + description, strip journalist
     boilerplate from the description.
  2. Extract dense [CLS] embeddings via ProsusAI/finbert (NO sentiment head).
  3. Apply PCA dimensionality reduction: 768-d → 16-d.
  4. Extract sentiment scores via ProsusAI/finbert sentiment classification head.
  5. Aggregate to one (16,) tensor per calendar day (mean-pool multi-article
     days) and one mean sentiment score per day.
  6. Persist embeddings as dict[str, torch.Tensor] → data/daily_news_embeddings.pt
     Persist sentiment as CSV → data/daily_news_sentiment.csv

Usage
─────
    python preprocess_news_embeddings.py            # uses GPU if available
    python preprocess_news_embeddings.py --cpu       # force CPU
    python preprocess_news_embeddings.py --batch 16  # override batch size
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.decomposition import PCA

# ─── constants ──────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent / "data"
INPUT_CSV  = DATA_DIR / "wheat_news.csv"
OUTPUT_PT  = DATA_DIR / "daily_news_embeddings.pt"
OUTPUT_CSV = DATA_DIR / "daily_news_sentiment.csv"

# FinBERT base model (NOT the sentiment-classification head)
MODEL_NAME = "ProsusAI/finbert"
MAX_LEN    = 512          # FinBERT max sequence length
BATCH_SIZE = 8            # default batch size; tune for your GPU RAM

# PCA dimensionality reduction
PCA_DIM    = 16           # reduce from 768-d to 16-d

# ─── regex: journalist boilerplate stripper ──────────────────────────────────
#
# Captures patterns like:
#   "By John Doe CHICAGO, March 27 (Reuters) - "
#   "By Julie Ingwersen MANHATTAN, Kansas, March 25 (Reuters) - "
#   "By Michael Hogan HAMBURG, March 19 (Reuters) - "
#   "* Wheat futures rise ..."  (Reuters bullet-point lead-ins)
#
# Strategy: match "By <words> <CITY>(, <State/Country>)? <rest> (<Source>) - "
# We also strip leading bullet-point markers ('*', '·', '•').
# ────────────────────────────────────────────────────────────────────────────
_BOILERPLATE_RE = re.compile(
    r"^"
    r"(?:\*\s*)?"                               # optional leading bullet
    r"(?:"
        r"By\s+[A-Z][a-zA-Z\s\-']+"            # "By <Reporter Name>"
        r"[A-Z]{2,}[\w\s,]*"                    # "CHICAGO, March 27" (CITY + date etc.)
        r"\([^)]+\)"                             # "(Reuters)" or "(Refinitiv)"
        r"\s*[-–—]\s*"                           # dash separator
    r")?"
    r"(?:\*\s*)?",                              # optional second bullet
    re.MULTILINE,
)


# ──────────────────────────────────────────────────────────────────────────────
#  1. TEXT CLEANING & CONCATENATION
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(row: pd.Series) -> str:
    """
    Concatenate title + description into `clean_text`, stripping journalist
    boilerplate from the description prefix.

    Many descriptions are truncated with "..." — by prepending the title we
    recover more semantic signal even when the description is partial.
    """
    title       = str(row.get("title", "")).strip()
    description = str(row.get("description", "")).strip()

    # Strip leading boilerplate from description
    description = _BOILERPLATE_RE.sub("", description).strip()

    # Remove trailing ellipsis artefact left by scraper truncation
    if description.endswith("..."):
        description = description[:-3].strip()

    # Concatenate: "Title. Description"
    if description:
        return f"{title}. {description}"
    return title


# ──────────────────────────────────────────────────────────────────────────────
#  2. EMBEDDING EXTRACTION  ([CLS] hidden state, 768-d)
# ──────────────────────────────────────────────────────────────────────────────

def extract_cls_embeddings(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:
    """
    Pass *texts* through FinBERT and return the [CLS] token's last hidden
    state for every input.  Shape: (len(texts), 768).

    All inference runs inside `torch.no_grad()` to save memory.
    """
    all_embeddings: list[torch.Tensor] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[start : start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        # Move every tensor in the batch to the target device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # outputs.last_hidden_state → (batch, seq_len, 768)
        # [CLS] token is always at position 0
        cls_embeddings = outputs.last_hidden_state[:, 0, :]   # (batch, 768)

        # Move back to CPU immediately to free GPU memory
        all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)           # (N, 768)


# ──────────────────────────────────────────────────────────────────────────────
#  2a. PCA DIMENSIONALITY REDUCTION  (768-d → 16-d)
# ──────────────────────────────────────────────────────────────────────────────

def apply_pca(embeddings: torch.Tensor, n_components: int = PCA_DIM) -> torch.Tensor:
    """
    Apply PCA to reduce embeddings from 768-d to n_components.
    
    Args:
        embeddings: Tensor of shape (N, 768)
        n_components: Target dimensionality (default 16)
    
    Returns:
        Reduced embeddings of shape (N, n_components)
    """
    print(f"[INFO] Applying PCA: {embeddings.shape[1]} → {n_components} dimensions")
    
    # Convert to numpy for scikit-learn
    embeddings_np = embeddings.numpy()
    
    # Fit and transform with PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings_np = pca.fit_transform(embeddings_np)
    
    # Explained variance
    explained_var_ratio = pca.explained_variance_ratio_.sum()
    print(f"[INFO] Explained variance ratio: {explained_var_ratio:.4f}")
    
    # Convert back to torch
    reduced_embeddings = torch.from_numpy(reduced_embeddings_np).float()
    
    return reduced_embeddings


# ──────────────────────────────────────────────────────────────────────────────
#  2b. SENTIMENT SCORING  (FinBERT classification head → positive/negative/neutral)
# ──────────────────────────────────────────────────────────────────────────────

def extract_sentiment_scores(
    texts: list[str],
    tokenizer: AutoTokenizer,
    sentiment_model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Run FinBERT *with* the sentiment classification head to produce per-article
    softmax probabilities for [positive, negative, neutral] and a scalar
    sentiment score (positive - negative).

    Returns a DataFrame with columns:
        sentiment_pos, sentiment_neg, sentiment_neu, sentiment_score
    """
    all_scores: list[dict] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Scoring sentiment"):
        batch_texts = texts[start : start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = sentiment_model(**encoded).logits          # (batch, 3)
            probs = torch.softmax(logits, dim=-1).cpu()         # (batch, 3)

        # FinBERT label order: positive=0, negative=1, neutral=2
        for row in probs:
            pos, neg, neu = row[0].item(), row[1].item(), row[2].item()
            all_scores.append({
                "sentiment_pos":   pos,
                "sentiment_neg":   neg,
                "sentiment_neu":   neu,
                "sentiment_score": pos - neg,
            })

    return pd.DataFrame(all_scores)


# ──────────────────────────────────────────────────────────────────────────────
#  3. TEMPORAL ALIGNMENT  (daily mean-pooling)
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_daily(
    df: pd.DataFrame,
    embeddings: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Group embeddings by calendar date (YYYY-MM-DD).  For dates with multiple
    articles, return the element-wise mean.  Returns a dict mapping each
    date string to a (768,) tensor.
    """
    # Ensure a clean date column (string form, day-level)
    dates = df["date_day"].tolist()

    daily_map: dict[str, list[int]] = {}
    for idx, d in enumerate(dates):
        daily_map.setdefault(d, []).append(idx)

    result: dict[str, torch.Tensor] = {}
    for day, indices in sorted(daily_map.items()):
        stacked = embeddings[indices]                 # (k, 768)
        result[day] = torch.mean(stacked, dim=0)      # (768,)

    return result


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FinBERT news → daily embeddings")
    parser.add_argument("--cpu",   action="store_true",  help="Force CPU inference")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default {BATCH_SIZE})")
    parser.add_argument("--input", type=str, default=str(INPUT_CSV),
                        help=f"Path to input CSV (default: {INPUT_CSV})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PT),
                        help=f"Path to output .pt file (default: {OUTPUT_PT})")
    args = parser.parse_args()

    # ── device ──────────────────────────────────────────────────────────────
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")          # Apple Silicon GPU
    else:
        device = torch.device("cpu")

    print(f"[INFO] Using device: {device}")

    # ── 1. load & clean ────────────────────────────────────────────────────
    print(f"[INFO] Loading news from {args.input}")
    df = pd.read_csv(args.input)
    print(f"[INFO] Loaded {len(df)} articles")

    # Convert date column → datetime → calendar day string
    df["date"]     = pd.to_datetime(df["date"])
    df["date_day"] = df["date"].dt.strftime("%Y-%m-%d")

    # Build clean_text column
    df["clean_text"] = df.apply(clean_text, axis=1)

    # Sanity preview
    print(f"\n[PREVIEW] First cleaned article:\n  {df['clean_text'].iloc[0][:200]}…\n")

    texts = df["clean_text"].tolist()

    # ── 2. load FinBERT ────────────────────────────────────────────────────
    print(f"[INFO] Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded → {device}")

    # ── 3. extract embeddings ──────────────────────────────────────────────
    embeddings = extract_cls_embeddings(texts, tokenizer, model, device, args.batch)
    print(f"[INFO] Embedding matrix shape: {embeddings.shape}")   # (N, 768)

    # ── 3a. apply PCA ──────────────────────────────────────────────────────
    embeddings = apply_pca(embeddings, PCA_DIM)
    print(f"[INFO] Reduced embedding matrix shape: {embeddings.shape}")   # (N, 16)

    # ── 3b. extract sentiment scores ─────────────────────────────────────
    print(f"[INFO] Loading FinBERT sentiment classification head")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_model.to(device)
    sentiment_model.eval()

    sent_df = extract_sentiment_scores(texts, tokenizer, sentiment_model, device, args.batch)
    df = pd.concat([df.reset_index(drop=True), sent_df], axis=1)
    print(f"[INFO] Sentiment scores: mean={df['sentiment_score'].mean():.4f}, "
          f"std={df['sentiment_score'].std():.4f}")

    del sentiment_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 4. daily aggregation ───────────────────────────────────────────────
    daily_embeddings = aggregate_daily(df, embeddings)
    n_days    = len(daily_embeddings)
    n_articles = len(df)
    print(f"[INFO] {n_articles} articles aggregated into {n_days} daily vectors")

    # Daily sentiment: mean-pool per-article scores by day
    daily_sent = (df.groupby("date_day")[["sentiment_pos", "sentiment_neg",
                                          "sentiment_neu", "sentiment_score"]]
                    .mean()
                    .reset_index()
                    .rename(columns={"date_day": "date"}))
    daily_sent["article_count"] = (df.groupby("date_day").size()
                                     .values)
    daily_sent = daily_sent.sort_values("date").reset_index(drop=True)

    # ── 5. save ────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    torch.save(daily_embeddings, output_path)
    print(f"[INFO] Saved daily embeddings → {output_path}")

    daily_sent.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved daily sentiment  → {OUTPUT_CSV}")

    # Quick verification
    sample_date = list(daily_embeddings.keys())[0]
    sample_vec  = daily_embeddings[sample_date]
    print(f"[CHECK] {sample_date} → tensor shape {sample_vec.shape}, "
          f"dtype {sample_vec.dtype}")

    # ── summary ────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  DONE — {n_days} daily embeddings ({PCA_DIM}-d after PCA)")
    print(f"       — {len(daily_sent)} daily sentiment scores")
    print(f"  Date range: {sorted(daily_embeddings.keys())[0]}"
          f" → {sorted(daily_embeddings.keys())[-1]}")
    print(f"  Outputs:    {output_path}")
    print(f"              {OUTPUT_CSV}")
    print("═" * 60)


if __name__ == "__main__":
    main()
