"""Restore narrowly defined quote typography at unique original-source offsets."""

# U+2018 is deliberately excluded: existing attribution/denial guards do not
# consistently recognize it as an apostrophe inside a word.
_TYPOGRAPHY = str.maketrans({'’': "'", '“': '"', '”': '"'})


def original_quote(text: str, quote: str) -> dict | None:
    """Return a unique literal span, changing only supported quote characters.

    Already exact citations keep their existing behavior. No words, whitespace,
    case, dashes, source identifiers, or semantic judgments are repaired here.
    """
    if not quote or quote in text or not any(char.isalnum() for char in quote):
        return None
    needle = quote.translate(_TYPOGRAPHY)
    haystack = text.translate(_TYPOGRAPHY)
    start = haystack.find(needle)
    if start < 0 or haystack.find(needle, start + 1) >= 0:
        return None
    # One code point maps to one code point; original character offsets hold.
    end = start + len(quote)
    return dict(quote=text[start:end], start=start, end=end)
