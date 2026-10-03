import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

DATA_RESUME = (
    "Data scientist skilled in Python, machine learning, scikit-learn, and Pandas. "
    "Built forecasting models from feature tables."
).encode()

REACT_RESUME = b"Frontend developer experienced with React and CSS for product interfaces."


def test_home_page_is_plain_language(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Investments that fit your resume" in response.text
    assert "See suggestions" in response.text
    assert "OpenAI API key" in response.text
    assert "not financial advice" in response.text
    assert "required_coverage" not in response.text
    assert "focus_coverage" not in response.text


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_fund_catalog_is_readable(client):
    response = client.get("/api/v1/investments")
    assert response.status_code == 200
    funds = response.json()
    assert len(funds) >= 12
    assert any(fund["id"] == "fund-ai" for fund in funds)


def test_data_science_resume_recommendation(client):
    response = client.post(
        "/api/v1/recommendations",
        files={"file": ("resume.txt", DATA_RESUME, "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    top = body["recommendations"][0]
    assert top["fund"]["id"] == "fund-ai"
    assert {"python", "machine learning", "scikit-learn"} <= set(top["matched_skills"])
    for field in ("focus_coverage", "related_coverage", "text_similarity", "score"):
        assert field in top
    assert "scikit-learn" in body["extracted_skills"]
    assert body["resume_years"] is None


def test_resume_years_are_returned(client):
    resume = (
        "Data scientist. Python, machine learning, and scikit-learn. "
        "6 years of experience. Harbor Systems 2019 - 2025."
    ).encode()
    response = client.post(
        "/api/v1/recommendations",
        files={"file": ("resume.txt", resume, "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["resume_years"] == 6
    assert body["recommendations"][0]["fund"]["ticker"] == "HAIF"


def test_openai_key_uses_extracted_years(client, monkeypatch):
    async def fake_extract(resume_text, lexicon, api_key):
        assert api_key == "sk-test-key"
        return ["python", "machine learning"], 1.8

    monkeypatch.setattr("app.api.recommendations.extract_with_openai", fake_extract)
    response = client.post(
        "/api/v1/recommendations",
        data={"openai_api_key": "sk-test-key"},
        files={"file": ("resume.txt", b"Python and machine learning.", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["resume_years"] == 1.8
    assert body["extracted_skills"] == ["python", "machine learning"]
    assert "sk-test-key" not in response.text


def test_react_resume_recommendation(client):
    response = client.post(
        "/api/v1/recommendations?top_k=3",
        files={"file": ("resume.txt", REACT_RESUME, "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 3
    assert body["recommendations"][0]["fund"]["id"] == "fund-software"


def test_rejects_unknown_extension(client):
    response = client.post(
        "/api/v1/recommendations",
        files={"file": ("resume.doc", b"Python", "application/msword")},
    )
    assert response.status_code == 400


def test_rejects_empty_text(client):
    response = client.post(
        "/api/v1/recommendations",
        files={"file": ("resume.txt", b"   \n", "text/plain")},
    )
    assert response.status_code == 400


def test_rejects_large_file(client):
    payload = b"a" * (2 * 1024 * 1024 + 1)
    response = client.post(
        "/api/v1/recommendations",
        files={"file": ("resume.txt", payload, "text/plain")},
    )
    assert response.status_code == 400
