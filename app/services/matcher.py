import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import Job, Recommendation, RecommendationResponse
from app.services.experience import experience_fit, extract_years
from app.services.skills import SkillExtractor, load_lexicon

# Skill coverage still answers whether the person can do the work.
# Experience fit answers whether the role asks for their years.
REQUIRED_WEIGHT = 0.50
PREFERRED_WEIGHT = 0.10
TEXT_WEIGHT = 0.20
EXPERIENCE_WEIGHT = 0.20

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_jobs(path: Path) -> list[Job]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Job.model_validate(item) for item in payload]


def _coverage(resume_skills: set[str], job_skills: list[str]) -> tuple[float, list[str], list[str]]:
    if not job_skills:
        return 0.0, [], []
    matched = [skill for skill in job_skills if skill in resume_skills]
    missing = [skill for skill in job_skills if skill not in resume_skills]
    return len(matched) / len(job_skills), matched, missing


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


class JobIndex:
    def __init__(self, jobs: list[Job], lexicon: list[str]) -> None:
        self.jobs = jobs
        self.extractor = SkillExtractor(lexicon)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.job_matrix = self.vectorizer.fit_transform([job.description for job in jobs])

    @classmethod
    def from_data_dir(cls, data_dir: Path = DATA_DIR) -> "JobIndex":
        jobs = load_jobs(data_dir / "jobs.json")
        lexicon = load_lexicon(data_dir / "skills.txt")
        return cls(jobs, lexicon)

    def rank(
        self,
        resume_text: str,
        top_k: int,
        location: str | None = None,
        seniority: str | None = None,
        skills: list[str] | None = None,
        years: float | None = None,
        use_given_years: bool = False,
    ) -> RecommendationResponse:
        extracted = self.extractor.extract(resume_text) if skills is None else skills
        resume_years = years if use_given_years else extract_years(resume_text)
        resume_skills = set(extracted)
        resume_vector = self.vectorizer.transform([resume_text])
        similarities = cosine_similarity(resume_vector, self.job_matrix)[0]

        ranked: list[Recommendation] = []
        for index, job in enumerate(self.jobs):
            if not _matches(job, location, seniority):
                continue
            required_coverage, required_matched, missing = _coverage(
                resume_skills, job.required_skills
            )
            preferred_coverage, preferred_matched, _ = _coverage(
                resume_skills, job.preferred_skills
            )
            text_similarity = float(max(0.0, similarities[index]))
            fit = experience_fit(resume_years, job.min_years, job.max_years)
            score = (
                REQUIRED_WEIGHT * required_coverage
                + PREFERRED_WEIGHT * preferred_coverage
                + TEXT_WEIGHT * text_similarity
                + EXPERIENCE_WEIGHT * fit
            )
            ranked.append(
                Recommendation(
                    job=job,
                    score=round(score, 4),
                    required_coverage=round(required_coverage, 4),
                    preferred_coverage=round(preferred_coverage, 4),
                    text_similarity=round(text_similarity, 4),
                    experience_fit=round(fit, 4),
                    matched_skills=_unique(required_matched + preferred_matched),
                    missing_skills=missing,
                )
            )

        ranked.sort(
            key=lambda item: (item.score, item.required_coverage, item.experience_fit),
            reverse=True,
        )
        return RecommendationResponse(
            extracted_skills=extracted,
            resume_years=resume_years,
            recommendations=ranked[:top_k],
        )


def _matches(job: Job, location: str | None, seniority: str | None) -> bool:
    if location and job.location.lower() != location.strip().lower():
        return False
    if seniority and job.seniority.lower() != seniority.strip().lower():
        return False
    return True
