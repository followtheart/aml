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


"""Match identical JSON tokens, preserving string values and scalar boundaries."""
import json,re
_PROFILE_JSON_KEY=re.compile(r'^\s*"(?:[^"\\]|\\.)+"\s*:')
_PROFILE_JSON_ATOM=re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|true|false|null')
def _profile_json_tokens(text):
    out=[];i=0;decoder=json.JSONDecoder()
    while i<len(text):
        if text[i] in ' \t\r\n':i+=1;continue
        start=i
        if text[i]=='"':
            try:value,end=decoder.raw_decode(text[i:])
            except ValueError:return None
            if not isinstance(value,str):return None
            i+=end
        elif text[i] in '{}[]:,':i+=1
        else:
            match=_PROFILE_JSON_ATOM.match(text,i)
            if not match:return None
            i=match.end()
            if i<len(text) and text[i] not in ' \t\r\n{}[]:,':return None
        out.append((text[start:i],start,i))
    return out

def original_profile_json_quote(source,quote):
    text=source.get('text','')
    if source.get('declared')!='persona' or not _PROFILE_JSON_KEY.match(text) or not _PROFILE_JSON_KEY.match(quote) or quote in text:return None
    source_tokens=_profile_json_tokens(text);quote_tokens=_profile_json_tokens(quote)
    if not source_tokens or not quote_tokens:return None
    needle=[t[0] for t in quote_tokens];haystack=[t[0] for t in source_tokens]
    matches=[i for i in range(len(haystack)-len(needle)+1) if haystack[i:i+len(needle)]==needle]
    if len(matches)!=1:return None
    i=matches[0];a,b=source_tokens[i][1],source_tokens[i+len(needle)-1][2]
    return dict(quote=text[a:b],start=a,end=b)
