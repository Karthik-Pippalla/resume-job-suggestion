from pathlib import Path

from app.schemas import Fund
from app.services.matcher import (
    FOCUS_WEIGHT,
    RELATED_WEIGHT,
    TEXT_WEIGHT,
    FundIndex,
    load_funds,
)
from app.services.skills import load_lexicon

DATA = Path(__file__).resolve().parent.parent / "app" / "data"


def _index() -> FundIndex:
    return FundIndex.from_data_dir(DATA)


def test_catalog_skills_are_in_the_lexicon():
    funds = load_funds(DATA / "funds.json")
    lexicon = set(load_lexicon(DATA / "skills.txt"))
    missing = {
        skill
        for fund in funds
        for skill in fund.focus_skills + fund.related_skills
        if skill not in lexicon
    }
    assert missing == set()


def test_data_science_resume_ranks_ai_above_energy():
    index = _index()
    resume = (
        "Data scientist skilled in Python, machine learning, scikit-learn, and Pandas. "
        "Built forecasting models from feature tables."
    )
    result = index.rank(resume, top_k=20)
    recommendations = result.recommendations
    ids = [item.fund.id for item in recommendations]
    assert ids.index("fund-ai") < ids.index("fund-energy")
    top = recommendations[0]
    assert top.fund.id == "fund-ai"
    assert {"python", "machine learning", "scikit-learn", "pandas"} <= set(top.matched_skills)
    assert top.focus_coverage == 1.0
    assert top.related_coverage > 0
    assert abs(
        FOCUS_WEIGHT * top.focus_coverage
        + RELATED_WEIGHT * top.related_coverage
        + TEXT_WEIGHT * top.text_similarity
        - top.score
    ) < 0.001


def test_react_resume_ranks_software_first():
    index = _index()
    resume = "Frontend developer experienced with React and CSS for product interfaces."
    recommendations = index.rank(resume, top_k=5).recommendations
    assert recommendations[0].fund.id == "fund-software"
    assert "react" in recommendations[0].matched_skills
    assert "css" in recommendations[0].matched_skills


def test_years_do_not_change_the_fund():
    index = _index()
    short = "Python, machine learning, and scikit-learn. 1 year of experience."
    long = "Python, machine learning, and scikit-learn. 8 years of experience."
    short_top = index.rank(short, top_k=1)
    long_top = index.rank(long, top_k=1)
    assert short_top.resume_years == 1
    assert long_top.resume_years == 8
    assert short_top.recommendations[0].fund.id == long_top.recommendations[0].fund.id == "fund-ai"


def test_empty_focus_skills_rely_on_text_similarity():
    broad = Fund(
        id="fund-open",
        name="Broad",
        ticker="BROD",
        sector="Broad market",
        risk="steadier",
        focus_skills=[],
        related_skills=[],
        description="Python machine learning models and scikit-learn research",
    )
    other = Fund(
        id="fund-other",
        name="Other",
        ticker="OTHR",
        sector="Energy",
        risk="more ups and downs",
        focus_skills=["figma"],
        related_skills=[],
        description="Oil wells and electric power plants",
    )
    index = FundIndex([broad, other], ["python", "machine learning", "scikit-learn", "figma"])
    recommendations = index.rank("Python machine learning scikit-learn", top_k=2).recommendations
    assert recommendations[0].fund.id == "fund-open"
    assert recommendations[0].focus_coverage == 0.0
    assert recommendations[0].text_similarity > recommendations[1].text_similarity
