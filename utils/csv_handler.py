"""CSV handling utilities for batch sentiment analysis."""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd

from src.sentiment_model import SentimentAnalyzer, SentimentResult


def process_csv_batch(
    file_obj,
    analyzer: SentimentAnalyzer,
    text_column: Optional[str] = None,
) -> pd.DataFrame:

    if not analyzer.is_loaded():
        raise RuntimeError("Model is not loaded. Call analyzer.load() first.")

    try:
        df = pd.read_csv(file_obj)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("CSV file is empty.")

    if text_column is not None:
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        col = text_column
    else:
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if not object_cols:
            raise ValueError(
                "No text column found. Please specify a column name or "
                "ensure the CSV has at least one text column."
            )
        col = object_cols[0]

    texts = df[col].fillna("").astype(str).tolist()
    results: list[SentimentResult] = analyzer.predict_batch(texts)

    output_df = pd.DataFrame(
        {
            "text": texts,
            "sentiment": [r.label for r in results],
            "confidence": [round(r.confidence * 100, 2) for r in results],
            "raw_label": [r.raw_label for r in results],
        }
    )

    return output_df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")