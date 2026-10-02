import json

import httpx

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
                    "years_of_experience is a number with one decimal, or null if the resume does not say. "
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
    return parse_extraction(content, lexicon)


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

    years = _years(payload.get("years_of_experience"))
    return skills, years


def _years(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        years = float(value)
    except (TypeError, ValueError):
        return None
    if years < 0 or years > 40:
        return None
    return round(years, 1)
