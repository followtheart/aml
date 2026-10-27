"""Optional lexical normalization for constraint candidate matching."""
import re
EXCEPTIONS = {'movies', 'cookies', 'brownies', 'zombies', 'smoothies', 'series', 'species', 'rookies', 'pies', 'ties', 'dies', 'lies'}
SPELLING = {'theatre': 'theater', 'theatres': 'theaters'}

def normalize(text):
    def token(m):
        word = m[0].lower()
        if word in SPELLING: return SPELLING[word]
        if len(word) > 4 and re.search(r'[^aeiou]ies$', word) and word not in EXCEPTIONS:
            return word[:-3] + 'y'
        return m[0]
    return re.sub(r"[A-Za-z]+", token, text)
