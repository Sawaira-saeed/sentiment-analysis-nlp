"""
Test suite for the Sentiment Analysis Classifier.

Tests cover:
    - Text preprocessing and validation.
    - Model loading and inference (positive, negative, neutral).
    - Empty and edge-case inputs.
    - Batch prediction.
    - CSV batch processing.

Run with: pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import clean_text, validate_input
from src.sentiment_model import get_analyzer, SentimentResult


# ─── Preprocessing Tests ───────────────────────────────────────────────

class TestPreprocessing:
    def test_clean_text_basic(self):
        assert clean_text("  Hello   World  ") == "Hello World"

    def test_clean_text_none(self):
        assert clean_text(None) == ""

    def test_clean_text_empty(self):
        assert clean_text("") == ""

    def test_clean_text_control_chars(self):
        assert clean_text("Hello\x00World") == "HelloWorld"

    def test_clean_text_multiple_spaces(self):
        assert clean_text("a    b\t\tc") == "a b c"

    def test_validate_input_valid(self):
        is_valid, msg = validate_input("Hello world")
        assert is_valid is True
        assert msg == ""

    def test_validate_input_empty(self):
        is_valid, msg = validate_input("")
        assert is_valid is False
        assert "empty" in msg.lower()

    def test_validate_input_none(self):
        is_valid, msg = validate_input(None)
        assert is_valid is False

    def test_validate_input_whitespace_only(self):
        is_valid, msg = validate_input("   ")
        assert is_valid is False

    def test_validate_input_too_long(self):
        is_valid, msg = validate_input("a" * 20000, max_length=10000)
        assert is_valid is False
        assert "too long" in msg.lower()


# ─── Model Tests ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def analyzer():
    """Load the model once for all tests in this module."""
    analyzer = get_analyzer()
    analyzer.load()
    return analyzer


class TestSentimentModel:
    def test_model_loaded(self, analyzer):
        assert analyzer.is_loaded() is True

    def test_positive_sentiment(self, analyzer):
        result = analyzer.predict(
            "I absolutely love this product! It's amazing and works perfectly."
        )
        assert isinstance(result, SentimentResult)
        assert result.label in ("POSITIVE", "NEGATIVE", "NEUTRAL")
        assert result.raw_label in ("POSITIVE", "NEGATIVE")
        assert 0.0 <= result.confidence <= 1.0
        assert result.raw_confidence > 0.5

    def test_negative_sentiment(self, analyzer):
        result = analyzer.predict(
            "This is terrible. I hate everything about it. Worst experience ever."
        )
        assert isinstance(result, SentimentResult)
        assert result.raw_label in ("POSITIVE", "NEGATIVE")
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_input_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.predict("")

    def test_none_input_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.predict(None)

    def test_long_text(self, analyzer):
        long_text = "This is a great product. " * 500
        result = analyzer.predict(long_text)
        assert isinstance(result, SentimentResult)
        assert result.label in ("POSITIVE", "NEGATIVE", "NEUTRAL")

    def test_probabilities_sum_to_one(self, analyzer):
        result = analyzer.predict("Good product")
        if result.probabilities:
            total = sum(result.probabilities.values())
            assert abs(total - 1.0) < 0.01

    def test_predict_batch(self, analyzer):
        texts = [
            "I love this!",
            "I hate this!",
            "",
            "It's okay I guess.",
        ]
        results = analyzer.predict_batch(texts)
        assert len(results) == 4
        assert all(isinstance(r, SentimentResult) for r in results)
        assert results[2].label == "NEUTRAL"

    def test_result_to_dict(self, analyzer):
        result = analyzer.predict("Great work!")
        d = result.to_dict()
        assert "label" in d
        assert "confidence" in d
        assert "probabilities" in d


# ─── CSV Handler Tests ─────────────────────────────────────────────────

class TestCSVHandler:
    def test_process_csv_batch(self, analyzer):
        csv_content = "text\nI love this product\nThis is terrible\nIt's okay\n"
        file_obj = io.StringIO(csv_content)
        from utils.csv_handler import process_csv_batch
        df = process_csv_batch(file_obj, analyzer)
        assert len(df) == 3
        assert "sentiment" in df.columns
        assert "confidence" in df.columns

    def test_csv_with_custom_column(self, analyzer):
        csv_content = "review,rating\nGreat product,5\nBad item,1\n"
        file_obj = io.StringIO(csv_content)
        from utils.csv_handler import process_csv_batch
        df = process_csv_batch(file_obj, analyzer, text_column="review")
        assert len(df) == 2

    def test_csv_empty_raises(self, analyzer):
        csv_content = "text\n"
        file_obj = io.StringIO(csv_content)
        from utils.csv_handler import process_csv_batch
        with pytest.raises(ValueError):
            process_csv_batch(file_obj, analyzer)

    def test_csv_invalid_column_raises(self, analyzer):
        csv_content = "text\nHello\n"
        file_obj = io.StringIO(csv_content)
        from utils.csv_handler import process_csv_batch
        with pytest.raises(ValueError):
            process_csv_batch(file_obj, analyzer, text_column="nonexistent")