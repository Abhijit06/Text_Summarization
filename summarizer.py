from transformers import pipeline, AutoTokenizer
from typing import Dict
import math

# ── Length presets (min_length, max_length) ──────────────────────────────────
LENGTH_PRESETS = {
    "short":  {"min_length": 30,  "max_length": 80},
    "medium": {"min_length": 80,  "max_length": 180},
    "long":   {"min_length": 150, "max_length": 10000},
}

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_MAP = {
    "bart": "facebook/bart-large-cnn",
    "t5":   "t5-small",
}

# ── Lazy-loaded pipelines (loaded once, reused) ───────────────────────────────
_pipelines: Dict[str, object] = {}


def _get_pipeline(model_name: str):
    model_id = MODEL_MAP.get(model_name, MODEL_MAP["bart"])
    if model_id not in _pipelines:
        print(f"[INFO] Loading model: {model_id} ...")
        _pipelines[model_id] = pipeline(
            "summarization",
            model=model_id,
            tokenizer=model_id,
        )
        print(f"[INFO] Model loaded: {model_id}")
    return _pipelines[model_id], model_id


# ── Chunking ──────────────────────────────────────────────────────────────────
MAX_TOKENS = 900   # BART max is 1024; keep headroom


def _chunk_text(text: str, model_id: str) -> list[str]:
    """Split text into chunks that fit within the model's token limit."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokens = tokenizer.encode(text, truncation=False)

    if len(tokens) <= MAX_TOKENS:
        return [text]

    # Split by sentences first, then group into chunks
    sentences = text.replace("\n", " ").split(". ")
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, truncation=False))
        if current_len + sentence_tokens > MAX_TOKENS and current:
            chunks.append(". ".join(current) + ".")
            current, current_len = [], 0
        current.append(sentence)
        current_len += sentence_tokens

    if current:
        chunks.append(". ".join(current))

    return chunks


# ── Main summarize function ───────────────────────────────────────────────────
def summarize_text(text: str, length: str = "medium", model_name: str = "bart") -> dict:
    length = length if length in LENGTH_PRESETS else "medium"
    params = LENGTH_PRESETS[length]

    pipe, model_id = _get_pipeline(model_name)
    chunks = _chunk_text(text.strip(), model_id)

    summaries = []
    for chunk in chunks:
        result = pipe(
            chunk,
            min_length=params["min_length"],
            max_length=params["max_length"],
            do_sample=False,
            truncation=True,
        )
        summaries.append(result[0]["summary_text"])

    # If multiple chunks, summarize the combined summaries once more
    final_summary = " ".join(summaries)
    if len(chunks) > 1 and len(final_summary.split()) > params["max_length"]:
        result = pipe(
            final_summary,
            min_length=params["min_length"],
            max_length=params["max_length"],
            do_sample=False,
            truncation=True,
        )
        final_summary = result[0]["summary_text"]

    return {
        "summary": final_summary,
        "model_used": model_id,
        "chunks_processed": len(chunks),
    }
