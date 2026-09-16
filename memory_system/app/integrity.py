"""Deterministic evidence and time constraints at the extraction boundary."""
import calendar
import re
import unicodedata
from datetime import datetime, timedelta, timezone

# Typographic variants small models silently "fix" when quoting.
_QUOTE_VARIANTS = str.maketrans({
    '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'",
    '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
    '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-', '\u2015': '-',
    '\u2026': '...', '\u00a0': ' ',
})
_EDGE_PUNCT = ' \t\r\n.,;:!?\'"-…()[]'


def normalize_text(value):
    """Formatting-insensitive form for containment checks; never changes words."""
    text = unicodedata.normalize('NFKC', str(value)).translate(_QUOTE_VARIANTS)
    return ' '.join(text.casefold().split())


def quote_in(quote, source):
    """True when the quote's words appear verbatim (modulo formatting) in source."""
    needle = normalize_text(quote).strip(_EDGE_PUNCT)
    return bool(needle) and needle in normalize_text(source)


def instant(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def validate_interval(start, end):
    if start:
        instant(start)
    if end:
        instant(end)
    if start and end and instant(end) < instant(start):
        raise ValueError('Validity interval ends before it starts')


def resolve_time(expression, reference):
    """Inclusive UTC ranges. Unknown expressions retain their text, never a guess."""
    raw = (expression or '').strip()
    result = dict(raw=raw or None, start=None, end=None, precision='unknown',
                  reference_time=reference)
    if not raw:
        return result
    text = raw.lower().strip(' .')
    ref = instant(reference) if reference else None
    day = ref.replace(hour=0, minute=0, second=0, microsecond=0) if ref else None
    start = end = None
    precision = 'unknown'
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
        try:
            start = instant(text)
            end = start + timedelta(days=1) - timedelta(microseconds=1)
            precision = 'day'
        except ValueError:
            pass
    elif re.fullmatch(r'\d{4}(-\d{2})?', text):
        try:
            year, month = int(text[:4]), int(text[5:]) if len(text) > 4 else 1
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            end = (datetime(year + 1, 1, 1, tzinfo=timezone.utc) if len(text) == 4
                   else start + timedelta(days=calendar.monthrange(year, month)[1]))
            end -= timedelta(microseconds=1)
            precision = 'year' if len(text) == 4 else 'month'
        except ValueError:
            pass
    elif re.fullmatch(r'\d{4}-\d{2}-\d{2}[tT].+', raw):
        try:
            start = end = instant(raw)
            precision = 'instant'
        except ValueError:
            pass
    elif day:
        offsets = {'yesterday': -1, 'today': 0, 'tomorrow': 1,
                   '昨天': -1, '今天': 0, '明天': 1}
        if text in offsets:
            start = day + timedelta(days=offsets[text])
            precision = 'day'
        weekdays = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6,
                    'mon': 0, 'tues': 1, 'tue': 1, 'wed': 2, 'thu': 3,
                    'thur': 3, 'fri': 4, 'sat': 5, 'sun': 6}
        match = re.fullmatch(r'(last|next) (' + '|'.join(weekdays) + r')', text)
        if match:
            target = weekdays[match[2]]
            delta = ((day.weekday() - target) % 7 or 7) if match[1] == 'last' else ((target - day.weekday()) % 7 or 7)
            start = day + timedelta(days=-delta if match[1] == 'last' else delta)
            precision = 'day'
        units = {'last week': ('week', -1), 'this week': ('week', 0), 'next week': ('week', 1),
                 'last month': ('month', -1), 'this month': ('month', 0), 'next month': ('month', 1),
                 'last year': ('year', -1), 'this year': ('year', 0), 'next year': ('year', 1),
                 '上周': ('week', -1), '本周': ('week', 0), '下周': ('week', 1),
                 '上个月': ('month', -1), '这个月': ('month', 0), '下个月': ('month', 1),
                 '去年': ('year', -1), '今年': ('year', 0), '明年': ('year', 1)}
        if text in units:
            precision, offset = units[text]
            if precision == 'week':
                start = day - timedelta(days=day.weekday()) + timedelta(weeks=offset)
                end = start + timedelta(days=7) - timedelta(microseconds=1)
            elif precision == 'month':
                total = day.year * 12 + day.month - 1 + offset
                year, month = total // 12, total % 12 + 1
                start = day.replace(year=year, month=month, day=1)
                end = start + timedelta(days=calendar.monthrange(year, month)[1]) - timedelta(microseconds=1)
            else:
                start = day.replace(year=day.year + offset, month=1, day=1)
                end = start.replace(year=start.year + 1) - timedelta(microseconds=1)
        if text in ('last weekend', 'this past weekend'):
            start = day - timedelta(days=(day.weekday() - 5) % 7 or 7)
            end = start + timedelta(days=2) - timedelta(microseconds=1)
            precision = 'weekend'
        words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
        match = re.fullmatch(r'(\d+|' + '|'.join(words) + r') days? ago', text)
        if match:
            start = day - timedelta(days=int(match[1]) if match[1].isdigit() else words[match[1]])
            precision = 'day'
        match = re.fullmatch(r'(\d+|' + '|'.join(words) + r') (months?|years?) ago', text)
        if match:
            count = int(match[1]) if match[1].isdigit() else words[match[1]]
            if match[2].startswith('year'):
                period = str(day.year - count)
            else:
                total = day.year * 12 + day.month - 1 - count
                period = f'{total // 12:04d}-{total % 12 + 1:02d}'
            resolved = resolve_time(period, reference)
            result.update(start=resolved['start'], end=resolved['end'], precision=resolved['precision'])
            return result
        match = re.fullmatch(r'(上|下)(?:周|星期)([一二三四五六日天])', text)
        if match:
            target = '一二三四五六日'.index(match[2].replace('天', '日'))
            start = day - timedelta(days=day.weekday()) + timedelta(days=target, weeks=-1 if match[1] == '上' else 1)
            precision = 'day'
        if start and precision == 'day' and end is None:
            end = start + timedelta(days=1) - timedelta(microseconds=1)
    if start and end:
        result.update(start=start.isoformat(), end=end.isoformat(), precision=precision)
    return result


# Only expressions resolve_time() can turn into a range; longest alternatives first.
_TIME_EXPRESSIONS = re.compile(
    r'\b(?:last weekend|this past weekend'
    r'|(?:last|next) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday'
    r'|mon|tues|tue|wed|thu|thur|fri|sat|sun)'
    r'|(?:last|this|next) (?:week|month|year)'
    r'|(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten) (?:days?|months?|years?) ago'
    r'|yesterday|today|tomorrow)\b'
    r'|[上下](?:周|星期)[一二三四五六日天]'
    r'|昨天|今天|明天|上周|本周|下周|上个月|这个月|下个月|去年|今年|明年', re.IGNORECASE)


def find_time_expressions(text):
    """Distinct resolvable relative time expressions present in text, in order."""
    found = []
    for match in _TIME_EXPRESSIONS.finditer(normalize_text(text)):
        if match[0] not in found:
            found.append(match[0])
    return found


def verify_quotes(fact, batch):
    evidence = fact.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Fact has no source evidence')
    for item in evidence:
        index, quote = item.get('message_index'), item.get('quote')
        if (type(index) is not int or not 0 <= index < len(batch)
                or not isinstance(quote, str) or not quote.strip()
                or not quote_in(quote, batch[index].content)):
            raise ValueError('Evidence quote does not match its source message')
    return sorted({item['message_index'] for item in evidence})


def cited_indices(fact, batch):
    """Best-effort sources of a fact whose evidence failed verification: the
    in-range indices it cites plus every message its quotes actually occur in
    (catches a right quote filed under a wrong index)."""
    evidence = fact.get('evidence') if isinstance(fact, dict) else None
    if not isinstance(evidence, list):
        return []
    found = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        index, quote = item.get('message_index'), item.get('quote')
        if type(index) is int and 0 <= index < len(batch):
            found.add(index)
        if isinstance(quote, str) and quote.strip():
            found.update(i for i, m in enumerate(batch) if quote_in(quote, m.content))
    return sorted(found)


def state_key(fact):
    """Only explicitly extracted single-valued states may supersede another state."""
    state = fact.get('state') or {}
    if fact.get('type') in ('event', 'episode'):
        return None
    if not all(isinstance(state.get(k), str) and state[k].strip()
               for k in ('subject', 'attribute', 'value')):
        return None
    return tuple(' '.join(state[k].casefold().split()) for k in ('subject', 'attribute'))


def may_supersede(old, new):
    key = state_key(new)
    return bool(key and key == state_key(old)
                and old['state']['value'].casefold() != new['state']['value'].casefold())
