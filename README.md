# Deep Research Agent — README

> A Streamlit UI that integrates a Gemini-powered deep-research pipeline with MongoDB user authentication and chat storage. Includes paper fetching (Semantic Scholar, arXiv, Crossref, SerpAPI, IEEE), embedding & ranking, report generation with Gemini, refinement loop (refine / ask / accept), and download/save functionality.

---

## Contents

- `app.py` — Main Streamlit application (integrates UI, auth, research pipeline).
- `.env` — Environment variables (not included; create this locally).
- `requirements.txt` — Python dependencies (included below and as a downloadable file).
- `README_and_requirements.md` — This document.

---

## Quick overview

1. Users sign up / sign in (stored in MongoDB).
2. Each user can create normal chats or _Research Chats_.
3. For a Research Chat, the user provides a topic (and optional parameters) and runs the pipeline:

   - Fetch papers from configured sources (Semantic Scholar, arXiv, Crossref, SerpAPI, IEEE when API keys provided).
   - Deduplicate, embed, score and rank papers.
   - Generate a comprehensive research report using Gemini (configurable model).
   - Save report and ranked papers to MongoDB and offer downloads (MD / CSV / JSON).

4. After generation, users can refine the report (the app classifies input into `refine`, `ask`, or `accept`), which will fetch more papers and regenerate the report or answer a question from the current report.

---

## Environment variables (`.env`)

Create a `.env` file at the project root with the following variables:

```env
MONGO_URI="your_mongo_connection_string"
MONGO_DBNAME="chat_app"  # optional, default used if omitted
GEMINI_API_KEY="your_gemini_api_key"
SEMANTIC_SCHOLAR_KEY="optional_semantic_scholar_api_key"
IEEE_API_KEY="optional_ieee_api_key"
SERPAPI_KEY="optional_serpapi_key"
```

**Notes:**

- `MONGO_URI` typically looks like `mongodb+srv://user:pass@cluster0.xyz.mongodb.net/?retryWrites=true&w=majority` when using Atlas.
- Do **not** commit `.env` or API keys to source control.

---

## Installation

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate    # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your `.env` file as shown above.

4. Run the app:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (usually [http://localhost:8501](http://localhost:8501)).

---

## Recommended workflow

1. Sign up and sign in.
2. Create a **New Research Chat** from the sidebar.
3. Provide a concise research Topic and optionally adjust settings in the `Research Settings` expander (max papers, top-K, weights, embedding/generation models).
4. Click **Run Research / Generate Report** and wait.
5. After generation, preview the report, download files, and use the `Refinement & Q&A` box for follow-ups.

---

## File storage and downloads

- Generated report and paper metadata are stored inside the `chats` collection in MongoDB (fields: `report_md`, `papers`, and `meta`).
- For convenience, the app also writes temporary files named `report_<chatid>.md|.csv|.json` on the server and exposes them via `st.download_button` so the user can download them through the browser.
- If you host the app on a server, downloaded files are served through the Streamlit session (the files still exist on the server filesystem). Consider switching to GridFS or S3 if you want centralized artifact storage.

---

## Notes about Gemini usage & costs

- The app uses Gemini both for embeddings and text generation. Both can incur API usage costs depending on your account and the models used.
- Embeddings are called for each paper during ranking; with many papers this can become slow and expensive.
- Consider using fewer `max_papers` for testing and gradually scale when you are satisfied.

---

## Troubleshooting

- `MONGO_URI` / connectivity errors: ensure your cluster allows connections from your IP and the connection string is correct.
- Gemini API problems: verify `GEMINI_API_KEY` and that the specified model name is available to your account.
- Missing optional API keys: the app will skip sources that are not configured (it will still run with arXiv / Crossref etc.).
- If feeds fail for arXiv, ensure `feedparser` is installed and that your environment allows outbound HTTP requests.

---

## Extending or customizing

- **Background jobs**: the app runs heavy tasks synchronously. If you need asynchronous/background generation, add a job queue (Redis + RQ / Celery) and expose job status via the UI.
- **Storage**: switch temporary file storage to GridFS or S3 for persistent file management.
- **Streaming**: implement streaming of the report generation if you want to show partial content as Gemini produces it.
- **Access control**: add roles (admin/team) or shareable team chats by augmenting the `chats` collection with `team_id` and ACL checks.

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Open a PR with a clear description of the change

---
