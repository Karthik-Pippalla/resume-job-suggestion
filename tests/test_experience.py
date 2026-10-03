from datetime import date

from app.services.experience import experience_fit, extract_years


def test_explicit_years_claim():
    assert extract_years("Backend engineer with 6 years of experience in Python.") == 6
    assert extract_years("1 year of experience.") == 1
    assert extract_years("3+ yrs building APIs.") == 3


def test_employment_dates_are_merged():
    text = "Acme 2016 - 2019\nNorthwind January 2019 - March 2022"
    assert extract_years(text, today=date(2026, 10, 2)) == 6.3


def test_present_role_uses_the_current_month():
    text = "Northwind Labs, January 2018 - Present"
    assert extract_years(text, today=date(2026, 10, 2)) == 8.8


def test_education_and_projects_are_not_work_experience():
    text = """
Summary
Jr Data Scientist with experience applying Python. GPA: 3.8
EDUCATION
Florida Atlantic University Master of Science Aug 2024 - May 2026
Experience
MasTec India LpL | Jr Data Scientist Jan 2023 - Aug 2024
PROJECTS
Ibhood 2026 - Present
"""
    assert extract_years(text, today=date(2026, 10, 2)) == 1.7


def test_explicit_claim_and_dates_are_not_added():
    text = "8 years of experience.\nHarbor Systems 2022 - Present"
    assert extract_years(text, today=date(2026, 10, 2)) == 8


def test_wrapped_job_dates_still_count():
    text = "Experience\nMasTec India\nJan 2023 -\nAug 2024"
    assert extract_years(text, today=date(2026, 10, 2)) == 1.7


def test_no_experience_signal_returns_none():
    assert extract_years("Python FastAPI SQL Docker") is None


def test_experience_fit_bands():
    assert experience_fit(1, 0, 2) == 1
    assert experience_fit(1, 5, None) == 0.2
    assert experience_fit(8, 5, None) == 1
    assert experience_fit(8, 2, 5) < 1
    assert experience_fit(None, 2, 5) == 0.5
