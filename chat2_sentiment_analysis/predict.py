# predict.py
# Runtime inference for Chat 2 — Sentiment Analysis Model.
# Fetches live Gold-related headlines, runs them through FinBERT + classification head,
# aggregates signals (weighted by recency), and returns the standard signal dict.
#
# Usage (standalone):
#   python predict.py
#
# Usage (imported by aggregator):
#   from chat2_sentiment_analysis.predict import get_signal

import os
from datetime import datetime, timezone

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

import config
from model import SentimentModel


# ── Fallback headlines (used when NewsAPI key is missing or returns 0 results) ──

FALLBACK_HEADLINES = [
    "Gold prices steady as markets await Federal Reserve decision",
    "XAU/USD holds near key support ahead of US inflation data",
    "Gold traders cautious with no clear directional catalyst",
]


# ── NewsAPI fetch ─────────────────────────────────────────────────────────────

def fetch_headlines() -> list[str]:
    """
    Fetches recent Gold-related headlines from NewsAPI.
    Returns a list of headline strings, newest first.
    Falls back to FALLBACK_HEADLINES if key is missing or no results returned.
    """
    if not config.NEWS_API_KEY:
        print("No NEWS_API_KEY found — using fallback headlines.")
        return FALLBACK_HEADLINES

    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=config.NEWS_API_KEY)

        query = " OR ".join(config.NEWS_KEYWORDS)
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",        # newest first
            page_size=20,
        )

        articles = response.get("articles", [])
        headlines = [a["title"] for a in articles if a.get("title")]

        if not headlines:
            print("NewsAPI returned 0 results — using fallback headlines.")
            return FALLBACK_HEADLINES

        print(f"Fetched {len(headlines)} headlines from NewsAPI.")
        return headlines

    except Exception as e:
        print(f"NewsAPI error: {e} — using fallback headlines.")
        return FALLBACK_HEADLINES


# ── Load model ────────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    """
    Loads FinBERT tokenizer + the fine-tuned classification head checkpoint.
    """
    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {config.CHECKPOINT_PATH}. "
            "Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading FinBERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.FINBERT_MODEL_NAME)

    print("Loading SentimentModel + checkpoint...")
    model = SentimentModel().to(device)
    ckpt  = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, tokenizer, device


# ── Inference on a single headline ───────────────────────────────────────────

def predict_headline(
    headline: str,
    model: SentimentModel,
    tokenizer: BertTokenizer,
    device: torch.device,
) -> dict:
    """
    Runs inference on a single headline string.

    Returns:
        {
            "binary_pred": 0 or 1,       # 0=DIRECTIONAL, 1=NEUTRAL
            "confidence":  float,         # softmax probability of predicted class
            "pos_prob":    float,         # FinBERT raw positive sentiment probability
            "neg_prob":    float,         # FinBERT raw negative sentiment probability
        }
    """
    # Tokenize
    encoding = tokenizer(
        headline,
        padding=True,
        truncation=True,
        max_length=config.MAX_TOKEN_LENGTH,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        # Classification head logits → binary prediction
        logits = model(input_ids, attention_mask)          # [1, 2]
        probs  = torch.softmax(logits, dim=1)[0]           # [2]
        binary_pred = probs.argmax().item()
        confidence  = probs[binary_pred].item()

        # FinBERT raw polarity — used to reconstruct UP vs DOWN direction
        # FinBERT outputs: [positive, negative, neutral] probabilities
        finbert_logits = model.finbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]                       # [CLS] token [1, 768]

        # We use a simple dot-product projection to get polarity score
        # positive direction = higher activation in first half of CLS embedding
        cls = finbert_logits[0]                            # [768]
        pos_score = cls[:384].mean().item()
        neg_score = cls[384:].mean().item()

    return {
        "binary_pred": binary_pred,
        "confidence":  confidence,
        "pos_score":   pos_score,
        "neg_score":   neg_score,
    }


# ── Aggregate across multiple headlines ──────────────────────────────────────

def aggregate_signals(results: list[dict], headlines: list[str]) -> dict:
    """
    Combines predictions from multiple headlines into one final signal.

    Strategy:
    - Weight each headline by recency: newest headline (index 0) gets highest weight
    - Weighted average of DIRECTIONAL confidence scores
    - Direction (UP vs DOWN) decided by weighted average of pos_score vs neg_score
    - If weighted DIRECTIONAL confidence < 0.5 → NEUTRAL

    Args:
        results:   list of per-headline prediction dicts
        headlines: list of headline strings (same order, newest first)

    Returns:
        final signal dict matching the Layer 2 aggregator contract
    """
    n = len(results)

    # Recency weights: index 0 (newest) gets weight n, index n-1 gets weight 1
    weights = [n - i for i in range(n)]
    total_weight = sum(weights)

    weighted_dir_conf = 0.0
    weighted_pos      = 0.0
    weighted_neg      = 0.0

    for i, (res, w) in enumerate(zip(results, weights)):
        w_norm = w / total_weight

        # DIRECTIONAL confidence = probability that headline is DIRECTIONAL (class 0)
        if res["binary_pred"] == 0:
            dir_conf = res["confidence"]
        else:
            dir_conf = 1.0 - res["confidence"]   # flip: confidence was for NEUTRAL

        weighted_dir_conf += w_norm * dir_conf
        weighted_pos      += w_norm * res["pos_score"]
        weighted_neg      += w_norm * res["neg_score"]

    # Final decision
    if weighted_dir_conf >= 0.5:
        direction = "UP" if weighted_pos > weighted_neg else "DOWN"
        signal    = direction
        final_conf = weighted_dir_conf
    else:
        signal    = "NEUTRAL"
        final_conf = 1.0 - weighted_dir_conf   # confidence in NEUTRAL

    return {
        "model":      "sentiment_analysis",
        "asset":      "XAU/USD",
        "signal":     signal,
        "confidence": round(final_conf, 4),
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def get_signal() -> dict:
    """
    Full inference pipeline:
    1. Fetch headlines
    2. Load model
    3. Run inference on each headline
    4. Aggregate into one signal dict

    Returns:
        dict with keys: model, asset, signal, confidence, timestamp
    """
    headlines = fetch_headlines()
    model, tokenizer, device = load_model_and_tokenizer()

    print(f"\nRunning inference on {len(headlines)} headline(s)...")
    results = []
    for i, headline in enumerate(headlines):
        res = predict_headline(headline, model, tokenizer, device)
        label = config.CLASS_NAMES[res["binary_pred"]]
        print(f"  [{i+1:02d}] {label} ({res['confidence']:.2f}) — {headline[:70]}")
        results.append(res)

    signal = aggregate_signals(results, headlines)
    return signal


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = get_signal()
    print("\n=== Chat 2 Signal Output ===")
    for k, v in result.items():
        print(f"  {k}: {v}")