import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import Fund, Recommendation, RecommendationResponse
from app.services.experience import extract_years
from app.services.skills import SkillExtractor, load_lexicon

# Focus skills answer whether the resume sits in the fund's field.
# Related skills and wording are smaller ties when the field is only partly clear.
FOCUS_WEIGHT = 0.65
RELATED_WEIGHT = 0.15
TEXT_WEIGHT = 0.20

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_funds(path: Path) -> list[Fund]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Fund.model_validate(item) for item in payload]


def _coverage(resume_skills: set[str], fund_skills: list[str]) -> tuple[float, list[str], list[str]]:
    if not fund_skills:
        return 0.0, [], []
    matched = [skill for skill in fund_skills if skill in resume_skills]
    missing = [skill for skill in fund_skills if skill not in resume_skills]
    return len(matched) / len(fund_skills), matched, missing


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


class FundIndex:
    def __init__(self, funds: list[Fund], lexicon: list[str]) -> None:
        self.funds = funds
        self.extractor = SkillExtractor(lexicon)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.fund_matrix = self.vectorizer.fit_transform([fund.description for fund in funds])

    @classmethod
    def from_data_dir(cls, data_dir: Path = DATA_DIR) -> "FundIndex":
        funds = load_funds(data_dir / "funds.json")
        lexicon = load_lexicon(data_dir / "skills.txt")
        return cls(funds, lexicon)

    def rank(
        self,
        resume_text: str,
        top_k: int,
        skills: list[str] | None = None,
        years: float | None = None,
        use_given_years: bool = False,
    ) -> RecommendationResponse:
        extracted = self.extractor.extract(resume_text) if skills is None else skills
        resume_years = years if use_given_years else extract_years(resume_text)
        resume_skills = set(extracted)
        resume_vector = self.vectorizer.transform([resume_text])
        similarities = cosine_similarity(resume_vector, self.fund_matrix)[0]

        ranked: list[Recommendation] = []
        for index, fund in enumerate(self.funds):
            focus_coverage, focus_matched, missing = _coverage(resume_skills, fund.focus_skills)
            related_coverage, related_matched, _ = _coverage(resume_skills, fund.related_skills)
            text_similarity = float(max(0.0, similarities[index]))
            score = (
                FOCUS_WEIGHT * focus_coverage
                + RELATED_WEIGHT * related_coverage
                + TEXT_WEIGHT * text_similarity
            )
            ranked.append(
                Recommendation(
                    fund=fund,
                    score=round(score, 4),
                    focus_coverage=round(focus_coverage, 4),
                    related_coverage=round(related_coverage, 4),
                    text_similarity=round(text_similarity, 4),
                    matched_skills=_unique(focus_matched + related_matched),
                    missing_skills=missing,
                )
            )

        ranked.sort(
            key=lambda item: (item.score, item.focus_coverage),
            reverse=True,
        )
        return RecommendationResponse(
            extracted_skills=extracted,
            resume_years=resume_years,
            recommendations=ranked[:top_k],
        )
