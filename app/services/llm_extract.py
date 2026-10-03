import json
import re

import httpx

from app.services.experience import extract_years

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"
MAX_RESUME_CHARS = 20000


class OpenAIExtractError(Exception):
    pass


async def extract_with_openai(
    resume_text: str,
    lexicon: list[str],
    api_key: str,
) -> tuple[list[str], float | None]:
    """Ask OpenAI for skills and years. The key is sent once and not stored."""
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You read resumes and return JSON only. "
                    "Count paid work, internships, and jobs. Do not count school or projects as employment. "
                    "years_of_experience must be a number with one decimal, computed from job dates. "
                    "January 2023 to August 2024 is 1.7. "
                    "Do not return a sentence, and do not return null when job dates are present. "
                    "skills is an array chosen only from the allowed list. "
                    "Include a skill only when the resume clearly uses it."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Allowed skills:\n"
                    + ", ".join(lexicon)
                    + "\n\nReturn JSON with keys years_of_experience and skills.\n\nResume:\n"
                    + resume_text[:MAX_RESUME_CHARS]
                ),
            },
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                OPENAI_URL,
                headers={"Authorization": f"Bearer {api_key.strip()}"},
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise OpenAIExtractError("The resume could not be read with that API key.") from exc

    if response.status_code in {401, 403}:
        raise OpenAIExtractError("That API key was rejected. Check it and try again.")
    if response.status_code == 429:
        raise OpenAIExtractError("That API key is busy right now. Wait a moment and try again.")
    if response.status_code >= 400:
        raise OpenAIExtractError("The resume could not be read with that API key.")

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise OpenAIExtractError("The resume could not be read with that API key.") from exc
    skills, years = parse_extraction(content, lexicon)
    if years is None:
        years = extract_years(resume_text)
    return skills, years


def parse_extraction(content: str, lexicon: list[str]) -> tuple[list[str], float | None]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OpenAIExtractError("The resume could not be read with that API key.") from exc
    if not isinstance(payload, dict):
        raise OpenAIExtractError("The resume could not be read with that API key.")

    allowed = set(lexicon)
    raw_skills = payload.get("skills") or []
    if not isinstance(raw_skills, list):
        raw_skills = []
    found = {str(skill).strip().lower() for skill in raw_skills}
    skills = [skill for skill in lexicon if skill in found and skill in allowed]

    return skills, _years_from_payload(payload)


def _years_from_payload(payload: dict) -> float | None:
    for key in ("years_of_experience", "years", "experience_years", "total_years"):
        years = _years(payload.get(key))
        if years is not None:
            return years
    nested = payload.get("experience")
    if isinstance(nested, dict):
        return _years(nested.get("years", nested.get("years_of_experience")))
    return None


_YEAR_PHRASE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:years|year|yrs|yr)\b(?:\s*(?:and|,)?\s*(\d+)\s*(?:months|month|mos)\b)?",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _years(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return _years(value.get("years", value.get("years_of_experience")))
    if isinstance(value, str):
        phrase = _YEAR_PHRASE.search(value)
        if phrase:
            years = float(phrase.group(1))
            if phrase.group(2):
                years += int(phrase.group(2)) / 12
            return _clamp_years(years)
        number = _NUMBER.search(value)
        if number is None:
            return None
        return _clamp_years(float(number.group(0)))
    try:
        return _clamp_years(float(value))
    except (TypeError, ValueError):
        return None


def _clamp_years(years: float) -> float | None:
    if years < 0 or years > 40:
        return None
    return round(years, 1)
