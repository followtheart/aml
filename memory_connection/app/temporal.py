"""Deterministic annotations of source time; never rewrite a source or infer facts."""
import re
import calendar
from datetime import datetime, timedelta, timezone


WEEKDAYS = {name: i for i, name in enumerate(
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"))}

NUMBERS = {word: i for i, word in enumerate(
    ('zero','one','two','three','four','five','six','seven','eight','nine','ten',
     'eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty'))}


def calendar_offsets(content, day):
    """Source-local calendar arithmetic with explicit precision; no fabricated day.

    Approximate language stays approximate. A month/year offset supplies that
    calendar period, never an exact event date or a claim that the event occurred.
    """
    number = r'(?:\d{1,3}|' + '|'.join(NUMBERS) + ')'
    pattern = rf'\b(?:(?:about|around|approximately|roughly)\s+)?(?:(?P<n>{number})\s+(?P<u>days?|weeks?|months?|years?)\s+ago|(?P<direction>last|next)\s+(?P<period>month|year))\b'
    result = []
    for m in re.finditer(pattern, content, re.I):
        # Do not silently parse the tail of an unsupported compound/range.
        prefix=content[:m.start()].lower()
        if re.search(r'[\d.,/+-]$',prefix):
            continue
        if re.search(r'(?:\b(?:'+ '|'.join(NUMBERS)+r'|or|to|between)|\d)[ -]+$',prefix):
            continue
        n = m.group('n')
        amount = (int(n) if n and n.isdigit() else NUMBERS[n.lower()] if n else 1)
        direction = 1 if (m.group('direction') or '').lower() == 'next' else -1
        unit = (m.group('u') or m.group('period')).lower().rstrip('s')
        try:
            if unit == 'year':
                year = day.year + direction*amount
                first, last = day.replace(year=year,month=1,day=1), day.replace(year=year,month=12,day=31)
                value = str(year)
            elif unit == 'month':
                year, month = divmod(day.year*12+day.month-1+direction*amount,12);month += 1
                first = day.replace(year=year,month=month,day=1)
                last = first.replace(day=calendar.monthrange(year,month)[1]);value = first.strftime('%Y-%m')
            else:
                first = last = day + timedelta(days=direction*amount*(7 if unit=='week' else 1))
                value = first.isoformat()
        except (ValueError,OverflowError):
            continue
        result.append({'phrase':m.group(),'start':m.start(),'end':m.end(),'anchor':day.isoformat(),
                       'from':first.isoformat(),'to':last.isoformat(),'resolved':value,'precision':unit,
                       'approximate':bool(re.match(r'(about|around|approximately|roughly)\b',m.group(),re.I)),
                       'convention':'source-local calendar offset; annotation, not an asserted event date'})
    return result


def source_time(source):
    stamp = source.get("timestamp")
    if stamp is None:
        return {"stated_at": None, "relative_dates": []}
    dt = datetime.fromtimestamp(stamp / 1000, timezone.utc)
    day = dt.date()
    notes = calendar_offsets(source['content'], day)
    pattern = r"\b(?:last (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|week)|yesterday|today|tomorrow)\b"
    for match in re.finditer(pattern, source["content"], re.I):
        phrase = match.group().lower()
        if phrase.split()[-1] in WEEKDAYS:
            delta = (day.weekday() - WEEKDAYS[phrase.split()[-1]]) % 7 or 7
            first = last = day - timedelta(days=delta)
            convention = "most recent strictly preceding named weekday"
        elif phrase == "last weekend":
            # If stated on a weekend, 'last weekend' is ambiguous: do not invent a date.
            if day.weekday() >= 5:
                notes.append({"phrase": match.group(), "start": match.start(), "end": match.end(),
                              "ambiguous": True, "anchor": day.isoformat()})
                continue
            last = day - timedelta(days=day.weekday()+1)
            first = last - timedelta(days=1)
            convention = "preceding completed Saturday-Sunday"
        elif phrase == "last week":
            first = day - timedelta(days=day.weekday()+7)
            last = first + timedelta(days=6)
            convention = "previous calendar week; natural language may be approximate"
        else:
            first = last = day + timedelta(days={"yesterday": -1, "today": 0, "tomorrow": 1}[phrase])
            convention = "relative calendar day"
        notes.append({"phrase": match.group(), "start": match.start(), "end": match.end(),
                      "from": first.isoformat(), "to": last.isoformat(), "anchor": day.isoformat(),
                      "convention": convention})
    return {"stated_at": dt.isoformat(), "relative_dates": notes}


def source_annotations(sources):
    return [{"source_id": s["source_id"], "message_id": s["message_id"], **source_time(s)} for s in sources]
