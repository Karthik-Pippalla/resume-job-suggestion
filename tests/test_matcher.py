from pathlib import Path

from app.schemas import Job
from app.services.matcher import (
    EXPERIENCE_WEIGHT,
    PREFERRED_WEIGHT,
    REQUIRED_WEIGHT,
    TEXT_WEIGHT,
    JobIndex,
    load_jobs,
)
from app.services.skills import load_lexicon

DATA = Path(__file__).resolve().parent.parent / "app" / "data"


def _index() -> JobIndex:
    return JobIndex.from_data_dir(DATA)


def test_catalog_skills_are_in_the_lexicon():
    jobs = load_jobs(DATA / "jobs.json")
    lexicon = set(load_lexicon(DATA / "skills.txt"))
    missing = {
        skill
        for job in jobs
        for skill in job.required_skills + job.preferred_skills
        if skill not in lexicon
    }
    assert missing == set()


def test_python_resume_ranks_backend_above_frontend():
    index = _index()
    resume = (
        "Backend engineer skilled in Python, FastAPI, SQL, and Docker. "
        "Built HTTP APIs and PostgreSQL schemas."
    )
    result = index.rank(resume, top_k=50)
    assert result.resume_years is None
    recommendations = result.recommendations
    ids = [item.job.id for item in recommendations]
    assert ids.index("job-backend-python") < ids.index("job-frontend-react")
    top = recommendations[0]
    assert {"python", "fastapi", "sql", "docker"} <= set(top.matched_skills)
    assert top.required_coverage == 1.0
    assert top.preferred_coverage > 0
    assert top.text_similarity >= 0
    assert abs(
        REQUIRED_WEIGHT * top.required_coverage
        + PREFERRED_WEIGHT * top.preferred_coverage
        + TEXT_WEIGHT * top.text_similarity
        + EXPERIENCE_WEIGHT * top.experience_fit
        - top.score
    ) < 0.001


def test_react_resume_ranks_frontend_first():
    index = _index()
    resume = "Frontend developer experienced with React and CSS for product interfaces."
    recommendations = index.rank(resume, top_k=5).recommendations
    assert recommendations[0].job.id == "job-frontend-react"
    assert "react" in recommendations[0].matched_skills
    assert "css" in recommendations[0].matched_skills


def test_location_filter_excludes_other_cities():
    index = _index()
    resume = "Python FastAPI SQL Docker"
    recommendations = index.rank(resume, top_k=20, location="Austin").recommendations
    assert recommendations
    assert all(item.job.location == "Austin" for item in recommendations)


def test_empty_required_skills_rely_on_text_similarity():
    job = Job(
        id="job-open",
        title="Generalist",
        company="Example",
        location="Remote",
        seniority="mid",
        required_skills=[],
        preferred_skills=[],
        description="Python FastAPI services and SQL databases",
    )
    other = Job(
        id="job-other",
        title="Unrelated",
        company="Example",
        location="Remote",
        seniority="mid",
        required_skills=["figma"],
        preferred_skills=[],
        description="Illustration and brand workshops",
    )
    index = JobIndex([job, other], ["python", "fastapi", "sql", "figma"])
    recommendations = index.rank("Python FastAPI SQL", top_k=2).recommendations
    assert recommendations[0].job.id == "job-open"
    assert recommendations[0].required_coverage == 0.0
    assert recommendations[0].text_similarity > recommendations[1].text_similarity


def _role(job_id: str, min_years: float, max_years: float | None) -> Job:
    return Job(
        id=job_id,
        title=job_id,
        company="Example",
        location="Remote",
        seniority="mid",
        min_years=min_years,
        max_years=max_years,
        required_skills=["python", "fastapi", "sql"],
        preferred_skills=["docker"],
        description="Python FastAPI SQL Docker APIs",
    )


def test_years_of_experience_pick_the_matching_band():
    index = JobIndex(
        [
            _role("job-junior", 0, 2),
            _role("job-mid", 2, 5),
            _role("job-senior", 5, None),
        ],
        ["python", "fastapi", "sql", "docker"],
    )
    junior_resume = "Python FastAPI SQL Docker. 1 year of experience."
    senior_resume = "Python FastAPI SQL Docker. January 2018 - Present. 8 years of experience."
    assert index.rank(junior_resume, top_k=1).recommendations[0].job.id == "job-junior"
    senior = index.rank(senior_resume, top_k=1)
    assert senior.resume_years >= 8
    assert senior.recommendations[0].job.id == "job-senior"
