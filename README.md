# Resume investment suggestions

A FastAPI service that ranks a seeded catalog of sample funds against an uploaded resume. Matching is deterministic. The page explains each fund in plain language. This is a demo, not financial advice, and not a suggestion to buy.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` to upload a resume and read the fund ideas. An OpenAI API key is optional: if you paste one, that search uses it to read skills and years of work, then discards it. `http://127.0.0.1:8000/docs` is the technical API.

```bash
curl -F "file=@resume.txt" "http://127.0.0.1:8000/api/v1/recommendations?top_k=5"
```

Tests:

```bash
pytest
```

## Endpoints

- `POST /api/v1/recommendations` — multipart file (`.pdf`, `.docx`, or `.txt`). Query: `top_k` (default 5, max 20). Optional form field: `openai_api_key`.
- `GET /api/v1/investments` — the seeded sample funds.
- `GET /health` — liveness.

Uploads over 2 MB, unknown extensions, and files with no extractable text return `400`. The file is parsed in memory and discarded.

## Scoring

For each fund:

- `focus_coverage` = matched focus skills / focus skills (0 when the fund lists none)
- `related_coverage` = matched related skills / related skills (0 when the fund lists none)
- `text_similarity` = cosine similarity between the resume and the fund description

`score = 0.65 * focus_coverage + 0.15 * related_coverage + 0.20 * text_similarity`

Years of work are read and returned as `resume_years`, but they do not change the ranking. Fund choice follows the field in the resume.

Years come from phrases such as "6 years" and from employment dates such as "Jan 2023 - Aug 2024". When the resume has an Experience heading, only that section is used, so education and project dates are left out.

Results are sorted by score, then focus coverage. Each suggestion includes `matched_skills` and `missing_skills`.

The TF-IDF model is fit once on fund descriptions at startup. The resume is only transformed. Skills come from `app/data/skills.txt` using token-boundary matches, so the same resume always yields the same skills.

## Decision log

- **Sample funds, not live market data.** The catalog in `app/data/funds.json` is fictional and loaded into memory. There is no broker, no prices, and no buy path.
- **Field match, not seniority.** Focus-skill coverage is 65% because the question is whether the resume sits in that fund's field. Related skills add 15%. TF-IDF cosine similarity is 20% so domain language still counts when the skill list is thin. Years of work are shown and then ignored for ranking.
- **Curated lexicon, with an optional OpenAI read.** Without a key, skills come from the lexicon. With a key, that one search asks OpenAI for skills and years, then discards the key.
- **Uploads are not stored.** A suggestion call does not need a resume database.
- **No authentication.** This is a local demo. Auth would matter before any shared deployment.
