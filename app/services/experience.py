import re
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

# Claims above this are not treated as career length (page numbers, old copyrights).
MAX_YEARS = 40
# Missing dates should not erase a skill match, and should not invent a senior profile.
UNKNOWN_EXPERIENCE_FIT = 0.5

_EXPLICIT_YEARS = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)\b",
    re.IGNORECASE,
)
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_END = r"(?P<end>(?:19|20)\d{2}|present|current|now)"
_RANGE = re.compile(
    rf"(?P<start>(?:19|20)\d{{2}})\s*(?:-|–|—|\bto\b)\s*{_END}\b",
    re.IGNORECASE,
)
_MONTH_RANGE = re.compile(
    rf"(?P<start_month>{_MONTH})[a-z.]*\s+(?P<start>(?:19|20)\d{{2}})"
    rf"\s*(?:-|–|—|\bto\b)\s*(?:(?P<end_month>{_MONTH})[a-z.]*\s+)?"
    rf"{_END}\b",
    re.IGNORECASE,
)
_PRESENT = {"present", "current", "now"}
_MONTH_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
# Own-line headings. Education and projects are not employment.
_HEADINGS = {
    "summary": "summary",
    "education": "education",
    "skills": "skills",
    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "employment": "experience",
    "projects": "projects",
    "publications": "publications",
    "certifications": "other",
}


def extract_years(text: str, today: date | None = None) -> float | None:
    """Years of work experience from claims and employment dates.

    Dates are read from the Experience section when that heading exists, so a
    degree ("Aug 2024 - May 2026") or a project ("2026 - Present") is not
    added to the job history. Months are counted when the resume includes
    them. An explicit claim such as "6 years" is not added on top of the
    dates; the larger of the two is used.
    """
    today = today or date.today()
    sections = _sections(text)
    claim_text = _claim_text(text, sections)
    date_text = sections.get("experience", text)
    explicit = [
        float(match)
        for match in _EXPLICIT_YEARS.findall(claim_text)
        if 0 < float(match) <= MAX_YEARS
    ]
    ranges = _employment_ranges(date_text, today)
    span = _merged_years(ranges)
    found = [value for value in (max(explicit) if explicit else None, span) if value is not None]
    if not found:
        return None
    return _round_years(min(max(found), float(MAX_YEARS)))


def experience_fit(years: float | None, min_years: float, max_years: float | None) -> float:
    """1 when the candidate sits in the job's experience band, lower outside it."""
    if years is None:
        return UNKNOWN_EXPERIENCE_FIT
    if years < min_years:
        if min_years <= 0:
            return 1.0
        return round(max(0.0, years / min_years), 4)
    if max_years is not None and years > max_years:
        over = years - max_years
        return round(max(0.0, 1.0 / (1.0 + over / 3.0)), 4)
    return 1.0


def _sections(text: str) -> dict[str, str]:
    current = "preamble"
    buckets: dict[str, list[str]] = {current: []}
    for line in text.splitlines():
        heading = _HEADINGS.get(line.strip().lower().rstrip(":"))
        if heading:
            current = heading
            buckets.setdefault(current, [])
            continue
        buckets.setdefault(current, []).append(line)
    return {name: "\n".join(lines) for name, lines in buckets.items() if lines}


def _claim_text(text: str, sections: dict[str, str]) -> str:
    if "experience" not in sections and "summary" not in sections:
        return text
    return "\n".join(sections[name] for name in ("summary", "experience") if name in sections)


def _employment_ranges(text: str, today: date) -> list[tuple[int, int]]:
    """Inclusive start month index and exclusive end month index."""
    ranges: list[tuple[int, int]] = []
    for match in _MONTH_RANGE.finditer(text):
        bounds = _bounds(
            int(match.group("start")),
            _month_number(match.group("start_month")),
            match.group("end"),
            _month_number(match.group("end_month")) if match.group("end_month") else None,
            today,
        )
        if bounds:
            ranges.append(bounds)
    for match in _RANGE.finditer(text):
        if _MONTH_RANGE.search(match.group(0)):
            continue
        bounds = _bounds(int(match.group("start")), 1, match.group("end"), None, today)
        if bounds:
            ranges.append(bounds)
    return ranges


def _bounds(
    start_year: int,
    start_month: int,
    end_token: str,
    end_month: int | None,
    today: date,
) -> tuple[int, int] | None:
    token = end_token.lower()
    if token in _PRESENT:
        end_year = today.year
        # The current month counts. Year-only "2022 - Present" starts in January
        # and runs through this month.
        resolved_end_month = today.month if end_month is None else end_month
        end_exclusive = _month_index(end_year, resolved_end_month) + 1
    else:
        end_year = int(token)
        if end_month is None:
            # "2016 - 2019" is a three-year span ending at the start of 2019.
            end_exclusive = _month_index(end_year, 1)
        else:
            end_exclusive = _month_index(end_year, end_month) + 1
    start = _month_index(start_year, start_month)
    if end_exclusive <= start or (end_exclusive - start) / 12 > MAX_YEARS:
        return None
    return start, end_exclusive


def _merged_years(ranges: list[tuple[int, int]]) -> float | None:
    if not ranges:
        return None
    ordered = sorted(ranges)
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    months = sum(end - start for start, end in merged)
    return months / 12


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def _month_number(token: str) -> int:
    return _MONTH_NUMBER[token[:3].lower()]


def _round_years(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
