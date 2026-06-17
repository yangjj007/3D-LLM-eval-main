"""
Token utilities for mesh token encoding/decoding.

Consolidates token_to_words and mesh token parsing logic from main.py and app.py.
"""

from typing import List, Optional
import re


def token_to_words(token_list: List[int]) -> str:
    """
    Convert a list of VQVAE token indices into the ShapeLLM mesh token string format.

    Example output: "<mesh-start><mesh42><mesh771>...<mesh-end>"
    """
    mesh_str = "<mesh-start>"
    for idx in token_list:
        mesh_str += f"<mesh{idx}>"
    mesh_str += "<mesh-end>"
    return mesh_str


def parse_mesh_tokens(response_text: str) -> List[int]:
    """
    Extract mesh token indices from LLM-generated text containing <meshX> tokens.

    Handles both structured output (with <mesh-start>/<mesh-end>) and raw token sequences.

    Returns:
        List[int]: Extracted token indices (may be fewer than 1024).
    """
    pattern = re.compile(r"<mesh(\d+)>")
    matches = pattern.findall(response_text)
    return [int(m) for m in matches]


def pad_tokens(tokens: List[int], target_length: int = 1024) -> List[int]:
    """
    Pad a token list to the target length by repeating the last token.
    If already at or exceeding target_length, truncate.
    """
    if len(tokens) == 0:
        return [0] * target_length
    while len(tokens) < target_length:
        tokens.append(tokens[-1])
    return tokens[:target_length]


def clean_response(response_text: str) -> str:
    """Remove special tokens from LLM response for display/metric computation."""
    return (
        response_text
        .replace("<|im_end|>", "")
        .replace("<|endoftext|>", "")
        .strip()
    )


def strip_mesh_tokens(response_text: str) -> str:
    """Remove all mesh token markup from response, leaving only natural language."""
    text = re.sub(r"<mesh-start>", "", response_text)
    text = re.sub(r"<mesh-end>", "", text)
    text = re.sub(r"<mesh\d+>", "", text)
    return text.strip()
