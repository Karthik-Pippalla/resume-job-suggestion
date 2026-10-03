import asyncio

import httpx

from app.services.llm_extract import OpenAIExtractError, extract_with_openai, parse_extraction


def test_parse_extraction_keeps_known_skills_and_years():
    content = '{"years_of_experience": 1.8, "skills": ["Python", "scikit-learn", "not-a-skill"]}'
    skills, years = parse_extraction(content, ["python", "sql", "scikit-learn"])
    assert skills == ["python", "scikit-learn"]
    assert years == 1.8


def test_parse_extraction_reads_a_year_phrase():
    _, years = parse_extraction(
        '{"years_of_experience": "1.7 years", "skills": ["python"]}',
        ["python"],
    )
    assert years == 1.7


def test_parse_extraction_reads_years_and_months():
    _, years = parse_extraction(
        '{"years": "1 year 8 months", "skills": []}',
        ["python"],
    )
    assert years == 1.7


def test_model_omitting_years_falls_back_to_resume_dates():
    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "choices": [
                {
                    "message": {
                        "content": '{"years_of_experience": null, "skills": ["python"]}',
                    }
                }
            ]
        }
        return httpx.Response(200, json=body)

    original = httpx.AsyncClient

    class Client:
        def __init__(self, *args, **kwargs):
            self._client = original(transport=httpx.MockTransport(handler))

        async def __aenter__(self):
            return self._client

        async def __aexit__(self, *args):
            await self._client.aclose()

    httpx.AsyncClient = Client
    try:
        skills, years = asyncio.run(
            extract_with_openai(
                "Experience\nMasTec Jan 2023 - Aug 2024\n6 years of experience in Python.",
                ["python"],
                "sk-test",
            )
        )
    finally:
        httpx.AsyncClient = original
    assert skills == ["python"]
    assert years == 6


def test_parse_extraction_allows_unknown_years():
    skills, years = parse_extraction('{"years_of_experience": null, "skills": []}', ["python"])
    assert skills == []
    assert years is None


def test_rejected_api_key_does_not_echo_the_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-secret"
        return httpx.Response(401, json={"error": {"message": "invalid_api_key sk-secret"}})

    original = httpx.AsyncClient

    class Client:
        def __init__(self, *args, **kwargs):
            self._client = original(transport=httpx.MockTransport(handler))

        async def __aenter__(self):
            return self._client

        async def __aexit__(self, *args):
            await self._client.aclose()

    httpx.AsyncClient = Client
    try:
        try:
            asyncio.run(extract_with_openai("Python resume", ["python"], "sk-secret"))
        except OpenAIExtractError as exc:
            assert "sk-secret" not in str(exc)
            assert "rejected" in str(exc)
        else:
            raise AssertionError("expected the key to be rejected")
    finally:
        httpx.AsyncClient = original
