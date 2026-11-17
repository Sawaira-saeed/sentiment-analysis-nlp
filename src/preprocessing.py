"""
Text preprocessing utilities for sentiment analysis.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean and normalize input text for sentiment analysis.

    Steps:
        - Strip leading/trailing whitespace.
        - Normalize unicode characters to NFC form.
        - Collapse multiple whitespace into single spaces.
        - Remove control characters.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string. Returns empty string if input is None or empty.
    """
    if text is None:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def validate_input(text: str, min_length: int = 1, max_length: int = 10000) -> tuple[bool, str]:
    """
    Validate user text input.

    Args:
        text: Input string to validate.
        min_length: Minimum allowed character count.
        max_length: Maximum allowed character count.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty when valid.
    """
    if text is None or not str(text).strip():
        return False, "Input text is empty. Please enter some text to analyze."

    text = str(text).strip()
    if len(text) < min_length:
        return False, f"Input is too short. Minimum {min_length} character(s) required."

    if len(text) > max_length:
        return False, f"Input is too long. Maximum {max_length} characters allowed."

    return True, ""