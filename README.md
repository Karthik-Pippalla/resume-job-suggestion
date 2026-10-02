# Resume Job Suggestion API

A FastAPI service that ranks a seeded job catalog against an uploaded resume. Matching is deterministic and the score is broken into parts so a reviewer can see why a job ranked where it did.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` to upload a resume and read the suggestions in plain language. An OpenAI API key is optional: if you paste one, that search uses it to read skills and years of work, then discards it. `http://127.0.0.1:8000/docs` is the technical API.

```bash
curl -F "file=@resume.txt" "http://127.0.0.1:8000/api/v1/recommendations?top_k=5"
```

Tests:

```bash
pytest
```

## Endpoints

- `POST /api/v1/recommendations` — multipart file (`.pdf`, `.docx`, or `.txt`). Query: `top_k` (default 5, max 20), optional `location`, optional `seniority`.
- `GET /api/v1/jobs` — the seeded catalog.
- `GET /health` — liveness.

Uploads over 2 MB, unknown extensions, and files with no extractable text return `400`. The file is parsed in memory and discarded.

## Scoring

For each job:

- `required_coverage` = matched required skills / required skills (0 when the job lists none)
- `preferred_coverage` = matched preferred skills / preferred skills (0 when the job lists none)
- `text_similarity` = cosine similarity between the resume and the job description
- `experience_fit` = 1 when the resume's years fall in the job's `min_years`–`max_years` band, lower when the candidate is under or over that band, and 0.5 when no experience could be read

`score = 0.50 * required_coverage + 0.10 * preferred_coverage + 0.20 * text_similarity + 0.20 * experience_fit`

Years come from phrases such as "6 years" and from employment dates such as "Jan 2023 - Aug 2024" or "January 2018 - Present". When the resume has an Experience heading, only that section is used, so education and project dates are left out. Overlapping jobs are merged, and a summary claim is not added on top of those dates. The response includes `resume_years` (`null` when nothing could be read).

Results are sorted by score, then required coverage, then experience fit. Each suggestion includes `matched_skills` and `missing_skills`.

The TF-IDF model is fit once on job descriptions at startup. The resume is only transformed. Skills come from `app/data/skills.txt` using token-boundary matches, so the same resume always yields the same skills.

## Decision log

- **Seeded JSON catalog, loaded into memory.** The catalog is small and read-only. A file in the repo is easier to audit than a database or a live jobs API, and there is no write path that would justify SQLite.
- **Fixed weights in `app/services/matcher.py`.** Required-skill coverage is 50% because the main question is whether the person can do the role. Preferred skills add 10%. TF-IDF cosine similarity is 20% so domain language still counts when skill lists are thin. Experience fit is 20% so a role that asks for the candidate's years ranks above an otherwise similar role that does not. The components are returned with every suggestion.
- **Experience is read from the resume text, not guessed by a model.** A year claim or employment dates set `resume_years`. Dates under Education or Projects are ignored when an Experience heading is present, and months are counted when the resume includes them. Each job in `app/data/jobs.json` declares `min_years` and `max_years` (junior 0–2, mid 2–5, senior 5+). If the resume has no dates and no year claim, every job gets a neutral experience fit so the skill match still stands.
- **Curated lexicon, not an LLM extractor.** A miss is a data fix. Rankings stay reproducible across runs and machines.
- **Uploads are not stored.** A suggestion call does not need a resume database, and keeping files would create a privacy problem this task did not ask for.
- **No authentication.** This is a local assessment service. Auth would matter before any shared deployment; it is out of scope here.
