"""Prompt template loader. Templates live in ../../prompts relative to repo
root (the aml/prompts directory shared with the design doc)."""
import re
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache(maxsize=16)
def load(name: str) -> str:
    text = (PROMPT_DIR / name).read_text(encoding="utf-8")
    # strip leading "#" comment header lines
    return "\n".join(l for l in text.splitlines()
                     if not l.startswith("#")).strip()


def render(name: str, **kwargs) -> str:
    return load(name).format(**kwargs)
