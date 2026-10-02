import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

PYTHON_RESUME = (
    "Backend engineer skilled in Python, FastAPI, SQL, and Docker. "
    "Built HTTP APIs and PostgreSQL schemas."
).encode()

REACT_RESUME = b"Frontend developer experienced with React and CSS for product interfaces."


def test_home_page_is_plain_language(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Jobs that fit your resume" in response.text
    assert "See suggestions" in response.text
    assert "OpenAI API key" in response.text
    assert "required_coverage" not in response.text


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_jobs_catalog_is_readable(client):
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert len(jobs) >= 24
    assert any(job["id"] == "job-backend-python" for job in jobs)


def test_python_resume_recommendation(client):
    response = client.post(
        "/api/v1/recommendations",
        files={"file": ("resume.txt", PYTHON_RESUME, "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    top = body["recommendations"][0]
    assert top["job"]["id"] == "job-backend-python"
    assert {"python", "fastapi", "sql", "docker"} <= set(top["matched_skills"])
    for field in (
        "required_coverage",
        "preferred_coverage",
        "text_similarity",
        "experience_fit",
        "score",
    ):
        assert field in top
    assert "fastapi" in body["extracted_skills"]
    assert body["resume_years"] is None


def test_resume_years_are_returned(client):
    resume = (
        "Backend engineer. Python, FastAPI, SQL, and Docker. "
        "6 years of experience. Harbor Systems 2019 - 2025."
    ).encode()
    response = client.post(
        "/api/v1/recommendations",
        files={"file": ("resume.txt", resume, "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["resume_years"] == 6
    assert "min_years" in body["recommendations"][0]["job"]


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
    assert body["recommendations"][0]["job"]["id"] == "job-frontend-react"


def test_seniority_filter(client):
    response = client.post(
        "/api/v1/recommendations?seniority=senior&top_k=20",
        files={"file": ("resume.txt", PYTHON_RESUME, "text/plain")},
    )
    assert response.status_code == 200
    recommendations = response.json()["recommendations"]
    assert recommendations
    assert all(item["job"]["seniority"] == "senior" for item in recommendations)


def test_rejects_unknown_extension(client):
    response = client.post(
        "/api/v1/recommendations",
        files={"file": ("resume.doc", b"Python FastAPI", "application/msword")},
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
