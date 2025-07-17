import json
import re
import os
from nltk.tokenize import word_tokenize

def make_intervening_regex(head: str, tail: str, max_gap: int = 6) -> re.Pattern:
    """
    Buat regex yang memperbolehkan hingga `max_gap` kata antara `head` dan `tail`.
    """
    pattern = rf"{re.escape(head)}(?:\s+\w+){{0,{max_gap}}}\s+{tail}"
    return re.compile(pattern, flags=re.IGNORECASE)

def load_booster_config(config_path: str):
    """
    Baca file JSON dan kembalikan:
      - token_pairs: list of (head, tail)
      - regex_patterns: list of compiled re.Pattern
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    token_pairs = [tuple(pair) for pair in cfg.get("token_booster_pairs", [])]

    regex_patterns = []
    for item in cfg.get("regex_booster_patterns", []):
        head = item["head"]
        tail = item["tail"]
        regex_patterns.append(make_intervening_regex(head, tail))

    return token_pairs, regex_patterns

def is_booster_aduan(text: str,
                        token_pairs: list[tuple[str,str]],
                        regex_patterns: list[re.Pattern],
                        window: int = 3):
    """
    Cek rule-based booster aduan:
    1) Token‐based: cari `head` di token list, periksa `tail` di jendela berikutnya.
    2) Regex‐based: cari pola gap-aware.
    """
    text_norm = re.sub(r"\s+", " ", text.lower().strip())
    tokens = word_tokenize(text_norm)

    # 1) Token‐based matching
    for head, tail in token_pairs:
        if head in tokens:
            for idx, w in enumerate(tokens):
                if w == head:
                    window_tokens = tokens[idx+1 : idx+1+window]
                    if tail in window_tokens:
                        return True, f"[TokenBooster: {head}→{tail}]"

    # 2) Regex‐based matching
    for pat in regex_patterns:
        if pat.search(text_norm):
            return True, f"[RegexBooster: {pat.pattern}]"

    return False, None
