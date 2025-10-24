# app.py
import os
import time
import uuid
import bcrypt
import json
import csv
import textwrap
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
import requests
from urllib.parse import quote

import google.generativeai as genai
import numpy as np

from typing import Optional

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from io import BytesIO
import markdown as md_lib
from bs4 import BeautifulSoup
import base64

# -----------------------
# Load env / config
# -----------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_NAME = os.getenv("MONGO_DBNAME", "chat_app")

SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY")
IEEE_API_KEY = os.getenv("IEEE_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable required")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    # show warning but allow app to load
    st.warning("GEMINI_API_KEY not set; Gemini calls will fail until you set it in environment.")

# -----------------------
# MongoDB setup
# -----------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_col = db["users"]
chats_col = db["chats"]

# -----------------------
# Helpers: auth + chat storage
# -----------------------
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)

def create_user(username: str, password: str) -> bool:
    if users_col.find_one({"username": username}):
        return False
    users_col.insert_one({"username": username, "password": hash_password(password)})
    return True

def authenticate_user(username: str, password: str) -> bool:
    user = users_col.find_one({"username": username})
    return bool(user and check_password(password, user["password"]))

def create_chat(username: str, title: str = "New Chat") -> dict:
    chat_id = str(uuid.uuid4())
    chat = {
        "_id": chat_id,
        "username": username,
        "title": title,
        "type": "research",
        "messages": [],
        "created_at": time.time(),
        "updated_at": time.time(),
        "report_md": None,
        "papers": [],           # papers used for full report (saved after generating report)
        "ranked_papers": [],    # serialized ranked papers stored when "Run Research" finishes
        "meta": {},
        "qa_history": []        # list of {"q":..., "a":...}
    }
    chats_col.insert_one(chat)
    return chat

def get_user_chats(username: str) -> List[dict]:
    return list(chats_col.find({"username": username}).sort("updated_at", -1))

def get_chat(chat_id: str) -> dict:
    return chats_col.find_one({"_id": chat_id})

def update_chat_report_and_papers(chat_id: str, report_md: str, papers: List[Dict[str, Any]], meta: Dict[str, Any] = None) -> None:
    chats_col.update_one({"_id": chat_id}, {"$set": {
        "report_md": report_md,
        "papers": papers,
        "meta": meta or {},
        "updated_at": time.time()
    }})

# New helpers for persistence of ranked papers and QA
def update_chat_ranked_papers(chat_id: str, ranked_papers: List[Dict[str, Any]], meta: Dict[str, Any] = None) -> None:
    """Save ranked papers (serializable dicts) to chat and update meta/updated_at."""
    # preserve existing meta keys and merge
    existing = chats_col.find_one({"_id": chat_id}) or {}
    existing_meta = existing.get("meta", {}) if isinstance(existing.get("meta", {}), dict) else {}
    merged_meta = {**existing_meta, **(meta or {})}
    chats_col.update_one({"_id": chat_id}, {"$set": {
        "ranked_papers": ranked_papers,
        "meta": merged_meta,
        "updated_at": time.time()
    }})

def append_chat_qa(chat_id: str, q: str, a: str) -> None:
    """Append one Q/A pair to the chat's qa_history and bump updated_at."""
    chats_col.update_one({"_id": chat_id}, {"$push": {"qa_history": {"q": q, "a": a}}, "$set": {"updated_at": time.time()}})

# -----------------------
# Paper dataclass & scoring helpers
# -----------------------
NOW_YEAR = time.gmtime().tm_year

@dataclass
class Paper:
    source: str
    paper_id: str
    title: str
    abstract: str
    year: Optional[int]
    authors: List[str]
    venue: Optional[str]
    url: Optional[str]
    pdf_url: Optional[str]
    doi: Optional[str]
    citation_count: Optional[int]
    similarity: float = 0.0
    score: float = 0.0

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)

def _get_json(url: str, params: dict = None, headers: dict = None, timeout: int = 30) -> dict:
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def _head(url: str, timeout: int = 8) -> bool:
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        return resp.ok
    except requests.RequestException:
        return False

def _log1p_normalized(x: Optional[int]) -> float:
    if not x or x <= 0:
        return 0.0
    return min(1.0, math.log1p(x) / math.log(1 + 1000))

def _recency_score(year: Optional[int]) -> float:
    if not year:
        return 0.0
    age = max(0, NOW_YEAR - int(year))
    return max(0.0, 1.0 - (age / 12.0))

# -----------------------
# Data source clients (adapted)
# -----------------------
class SemanticScholarClient:
    BASE = "https://api.semanticscholar.org/graph/v1"
    def __init__(self, api_key: Optional[str] = None):
        self.headers = {"Accept": "application/json"}
        if api_key:
            self.headers["x-api-key"] = api_key

    def search(self, query: str, limit: int = 50) -> List[Paper]:
        fields = ["title,abstract,year,authors,venue,externalIds,url,openAccessPdf,citationCount"]
        params = {"query": query, "limit": min(100, max(1, limit)), "fields": ",".join(fields)}
        res = _get_json(f"{self.BASE}/paper/search", params=params, headers=self.headers)
        data = res.get("data", []) if isinstance(res, dict) else []
        papers = []
        for p in data:
            authors = [a.get("name", "") for a in (p.get("authors") or [])]
            ext = p.get("externalIds") or {}
            doi = ext.get("DOI")
            pdf_url = (p.get("openAccessPdf") or {}).get("url")
            if pdf_url and not _head(pdf_url):
                pdf_url = None
            citation_count = p.get("citationCount")
            # try coerce to int if possible
            try:
                citation_count = int(citation_count) if citation_count is not None else None
            except Exception:
                citation_count = None
            papers.append(Paper(
                source="SemanticScholar",
                paper_id=str(p.get("paperId") or ""),
                title=p.get("title") or "",
                abstract=p.get("abstract") or "",
                year=p.get("year"),
                authors=authors,
                venue=p.get("venue"),
                url=p.get("url"),
                pdf_url=pdf_url,
                doi=doi,
                citation_count=citation_count,
            ))
        return papers

class ArxivClient:
    BASE = "http://export.arxiv.org/api/query"
    def search(self, query: str, limit: int = 50) -> List[Paper]:
        import feedparser
        encoded_query = quote(query)
        params = {"search_query": encoded_query, "start": 0, "max_results": min(50, max(1, limit))}
        url = f"{self.BASE}?search_query={params['search_query']}&start={params['start']}&max_results={params['max_results']}"
        feed = feedparser.parse(url)
        out = []
        for entry in (feed.entries or []):
            year = None
            try:
                year = int(entry.published.split("-")[0])
            except Exception:
                pass
            authors = [a.name for a in getattr(entry, "authors", [])]
            pdf_url = None
            link_url = None
            for l in getattr(entry, "links", []):
                if getattr(l, "rel", None) == "alternate":
                    link_url = getattr(l, "href", None)
                if getattr(l, "title", "") == "pdf":
                    pdf_url = getattr(l, "href", None)
            out.append(Paper(
                source="arXiv",
                paper_id=getattr(entry, "id", ""),
                title=getattr(entry, "title", ""),
                abstract=getattr(entry, "summary", ""),
                year=year,
                authors=authors,
                venue="arXiv",
                url=link_url,
                pdf_url=pdf_url,
                doi=None,
                citation_count=None,
            ))
        return out

class CrossrefClient:
    BASE = "https://api.crossref.org/works"
    def search(self, query: str, limit: int = 50) -> List[Paper]:
        params = {"query": query, "rows": min(50, max(1, limit))}
        res = _get_json(self.BASE, params=params)
        items = (res.get("message", {}).get("items") or [])
        out = []
        for it in items:
            year = None
            try:
                if "published-print" in it and "date-parts" in it["published-print"]:
                    year = it["published-print"]["date-parts"][0][0]
            except Exception:
                pass
            authors = []
            for a in it.get("author", []):
                authors.append(f"{a.get('given', '')} {a.get('family', '')}".strip())
            doi = it.get("DOI")
            url = it.get("URL")
            out.append(Paper(
                source="Crossref",
                paper_id=doi or url or "",
                title=(it.get("title", [""])[0] if it.get("title") else ""),
                abstract=it.get("abstract", "") or "",
                year=year,
                authors=authors,
                venue=(it.get("container-title", [None])[0]),
                url=url,
                pdf_url=None,
                doi=doi,
                citation_count=None,
            ))
        return out

class SerpAPIGoogleScholarClient:
    BASE = "https://serpapi.com/search.json"
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    def search(self, query: str, limit: int = 20) -> List[Paper]:
        if not self.api_key:
            return []
        params = {"engine": "google_scholar", "q": query, "api_key": self.api_key, "num": min(20, max(1, limit)), "hl": "en"}
        res = _get_json(self.BASE, params=params)
        results = (res.get("organic_results") or [])
        out = []
        for r in results:
            title = r.get("title") or ""
            url = r.get("link")
            snippet = r.get("snippet") or ""
            year = None
            pub_info = (r.get("publication_info") or {}).get("summary")
            if pub_info:
                for tok in pub_info.split():
                    if tok.isdigit() and len(tok) == 4:
                        try:
                            y = int(tok)
                            if 1900 < y <= NOW_YEAR:
                                year = y
                                break
                        except Exception:
                            pass
            citation_count = None
            if r.get("inline_links") and r["inline_links"].get("cited_by"):
                cited_str = r["inline_links"]["cited_by"].get("total")
                try:
                    citation_count = int(cited_str)
                except Exception:
                    citation_count = None
            out.append(Paper(
                source="GoogleScholar",
                paper_id=url or title,
                title=title,
                abstract=snippet,
                year=year,
                authors=[],
                venue=None,
                url=url,
                pdf_url=None,
                doi=None,
                citation_count=citation_count,
            ))
        return out

class IEEEXploreClient:
    BASE = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
    def search(self, query: str, limit: int = 30) -> List[Paper]:
        if not self.api_key:
            return []
        params = {
            "apikey": self.api_key,
            "format": "json",
            "max_records": min(200, max(1, limit)),
            "sort_order": "desc",
            "sort_field": "publication_year",
            "querytext": query,
        }
        res = _get_json(self.BASE, params=params)
        articles = (res.get("articles") or [])
        out = []
        for a in articles:
            year = None
            try:
                year = int(a.get("publication_year")) if a.get("publication_year") else None
            except Exception:
                year = None
            authors = []
            if a.get("authors") and a["authors"].get("authors"):
                authors = [au.get("full_name", "") for au in a["authors"]["authors"]]
            doi = a.get("doi")
            pdf_url = None
            if a.get("pdf_url"):
                pdf_url = a.get("pdf_url")
                if pdf_url and not _head(pdf_url):
                    pdf_url = None
            url = a.get("html_url") or a.get("pdf_url")
            citation_count = None
            try:
                citation_count = int((a.get("citing_paper_count") or 0))
            except Exception:
                citation_count = None
            out.append(Paper(
                source="IEEE",
                paper_id=a.get("article_number") or a.get("doi") or a.get("html_url") or "",
                title=a.get("title") or "",
                abstract=a.get("abstract") or "",
                year=year,
                authors=authors,
                venue=(a.get("publication_title") or a.get("publisher") or None),
                url=url,
                pdf_url=pdf_url,
                doi=doi,
                citation_count=citation_count,
            ))
        return out

# -----------------------
# Gemini embedding + helper functions
# -----------------------
def _setup_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai

def gemini_embed(embed_fn, text: str, task_type: str = "retrieval_query", model: str = "text-embedding-004") -> np.ndarray:
    res = embed_fn(model=model, content=text, task_type=task_type)
    vec = np.array(res["embedding"], dtype=np.float32)
    return vec

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# -----------------------
# Pipeline functions
# -----------------------
def fetch_papers(topic: str, limit: int, api_keys: Dict[str, str]) -> List[Paper]:
    collected: List[Paper] = []

    if api_keys.get("SEMANTIC_SCHOLAR_KEY"):
        try:
            sem = SemanticScholarClient(api_keys["SEMANTIC_SCHOLAR_KEY"])
            papers = sem.search(topic, limit)
            st.info(f"Retrieved {len(papers)} papers from Semantic Scholar")
            collected.extend(papers)
        except Exception as e:
            st.error(f"Semantic Scholar error: {e}")

    if api_keys.get("SERPAPI_KEY"):
        try:
            serp = SerpAPIGoogleScholarClient(api_keys["SERPAPI_KEY"])
            papers = serp.search(topic, limit // 3)
            st.info(f"Retrieved {len(papers)} papers from SerpAPI (Google Scholar)")
            collected.extend(papers)
        except Exception as e:
            st.error(f"SerpAPI error: {e}")

    try:
        arxiv_client = ArxivClient()
        papers = arxiv_client.search(topic, limit // 3)
        st.info(f"Retrieved {len(papers)} papers from arXiv")
        collected.extend(papers)
    except Exception as e:
        st.error(f"arXiv error: {e}")

    try:
        crossref_client = CrossrefClient()
        papers = crossref_client.search(topic, limit // 3)
        st.info(f"Retrieved {len(papers)} papers from Crossref")
        collected.extend(papers)
    except Exception as e:
        st.error(f"Crossref error: {e}")

    if api_keys.get("IEEE_KEY"):
        try:
            ieee_client = IEEEXploreClient(api_keys["IEEE_KEY"])
            papers = ieee_client.search(topic, limit)
            st.info(f"Retrieved {len(papers)} papers from IEEE Xplore")
            collected.extend(papers)
        except Exception as e:
            st.error(f"IEEE Xplore error: {e}")

    return collected

def exclude_papers(papers: List[Paper], exclude: List[Paper]) -> List[Paper]:
    exclude_keys = set((p.doi or f"{p.title.strip().lower()}::{p.year or ''}") for p in exclude)
    return [p for p in papers if (p.doi or f"{p.title.strip().lower()}::{p.year or ''}") not in exclude_keys]

def dedup_papers(collected: List[Paper]) -> List[Paper]:
    dedup: Dict[str, Paper] = {}
    for p in collected:
        key = (p.doi or f"{p.title.strip().lower()}::{p.year or ''}")
        if key not in dedup:
            dedup[key] = p
        else:
            q = dedup[key]
            for f in ["abstract", "venue", "url", "pdf_url", "citation_count"]:
                if getattr(q, f) in (None, "") and getattr(p, f) not in (None, ""):
                    setattr(q, f, getattr(p, f))
    return list(dedup.values())

def score_and_rank(papers: List[Paper], topic: str, weights: Tuple[float, float, float], gemini_api_key: str, embed_model: str = "text-embedding-004", model_name: str = "gemini-2.5-flash") -> List[Paper]:
    w_rel, w_cit, w_rec = weights
    gen = _setup_gemini(gemini_api_key)
    embed_fn = gen.embed_content

    topic_vec = gemini_embed(embed_fn, topic, task_type="retrieval_query", model=embed_model)

    progress = st.progress(0)
    out = []
    N = max(1, len(papers))
    for i, p in enumerate(papers):
        text = (p.title or "") + "\n\n" + (p.abstract or "")
        try:
            p_vec = gemini_embed(embed_fn, text[:7000], task_type="retrieval_document", model=embed_model)
            sim = cosine(topic_vec, p_vec)
            p.similarity = sim
        except Exception:
            p.similarity = 0.0
        c = _log1p_normalized(p.citation_count)
        r = _recency_score(p.year)
        p.score = w_rel * p.similarity + w_cit * c + w_rec * r
        out.append(p)
        progress.progress(min(100, int((i+1)/N*100)))
    out.sort(key=lambda x: x.score, reverse=True)
    return out

def build_context_chunks(papers: List[Paper], top_k: int) -> Tuple[str, str]:
    top = papers[:top_k]
    bibliography = []
    for i, p in enumerate(top, 1):
        authors = ", ".join(p.authors[:6]) + (" et al." if len(p.authors) > 6 else "")
        bib = f"[{i}] {p.title} — {authors} ({p.year or 'n.d.'}). {p.venue or ''}. DOI: {p.doi or 'N/A'}. Link: {p.url or p.pdf_url or 'N/A'}"
        bibliography.append(bib)

    context_chunks = []
    for idx, p in enumerate(top, 1):
        context_chunks.append(textwrap.dedent(f"""
        ### [{idx}] {p.title}
        - Venue: {p.venue or 'Unknown'} | Year: {p.year or 'n.d.'} | Citations: {p.citation_count or 0}
        - URL: {p.url or p.pdf_url or 'N/A'}
        - Abstract: {p.abstract or 'N/A'}
        """))

    return "\n".join(context_chunks), "\n".join(bibliography)

def generate_report(topic: str, papers: List[Paper], top_k: int, gemini_api_key: str, model_name: str = "gemini-2.5-flash") -> str:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name)
    context_str, bibliography_str = build_context_chunks(papers, top_k)

    sys_prompt = textwrap.dedent(f"""
    You are an expert research analyst. Given a topic and a set of top-ranked papers (with abstracts), write a THOROUGH, DETAILED, and COMPREHENSIVE research report with the following structure:
    1) Executive Summary (300-500 words)
    2) In-Depth Background & Core Concepts (400-600 words)
    3) Comparative Literature Synthesis (600-900 words)
    4) Critical Gap Analysis (300-500 words)
    5) Future Research Directions (300-500 words)
    6) Risks, Ethics, and Limitations (200-400 words)
    7) Practical Applications and Tooling Landscape (200-400 words)
    8) Conclusion (150-300 words)

    Use inline numeric citations like [1], [2] that map to the provided bibliography.
    """)
    user_prompt = f"Topic: {topic}\n\nTop Papers Context (ranked):\n" + context_str + "\n\nBibliography (use these citation indices):\n" + bibliography_str

    resp = model.generate_content(
        [
            {"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]},
        ],
        safety_settings=None,
        generation_config={"temperature": 0.6, "top_p": 0.9}
    )
    md = getattr(resp, "text", "")
    return md

def markdown_to_pdf_bytes(markdown_text: str) -> bytes:
    """Convert markdown text to styled PDF with bullet list support."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    code_style = ParagraphStyle('Code', parent=styles['Normal'], fontName='Courier', fontSize=9, leading=12, backColor='#f5f5f5')
    story = []

    # Convert markdown to HTML
    html = md_lib.markdown(markdown_text, extensions=['extra', 'codehilite', 'tables'])
    soup = BeautifulSoup(html, "html.parser")

    def handle_element(elem):
        if elem.name in ["h1", "h2", "h3"]:
            style = styles['Title'] if elem.name == "h1" else styles['Heading2'] if elem.name == "h2" else styles['Heading3']
            story.append(Paragraph(str(elem), style))
            story.append(Spacer(1, 10))
        elif elem.name == "p":
            story.append(Paragraph(str(elem), styles['Normal']))
            story.append(Spacer(1, 10))
        elif elem.name in ["ul", "ol"]:
            items = []
            for li in elem.find_all("li", recursive=False):
                items.append(ListItem(Paragraph(li.get_text(), styles['Normal'])))
            bulletType = 'bullet' if elem.name == "ul" else '1'
            story.append(ListFlowable(items, bulletType=bulletType, leftIndent=20))
            story.append(Spacer(1, 10))
        elif elem.name == "pre":
            code = elem.get_text()
            story.append(Paragraph(code, code_style))
            story.append(Spacer(1, 10))
        # Add more handlers as needed

    for elem in soup.body or soup:
        if getattr(elem, "name", None):
            handle_element(elem)

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

def answer_question(question: str, report_md: str, papers: List[Paper], top_k: int, gemini_api_key: str, model_name: str = "gemini-2.5-flash") -> str:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name)
    context_str, bibliography_str = build_context_chunks(papers, top_k)
    sys_prompt = textwrap.dedent("""
        You are an expert research assistant answering questions based on a set of academic papers.
        Your answers must:
        - stay true to the factual content of the papers and report. Cite papers with [#],
        - but can draw logical inferences or connect concepts *closely related* to the source material.
        If the question goes slightly beyond the paper, answer using plausible reasoning while noting that it’s an inferred connection.
    """)
    user_prompt = ("Question: " + question + "\n\n" + "Current Report (excerpt allowed):\n" + report_md[:200000] + "\n\n" + "Top Papers Context (ranked):\n" + context_str + "\n\nBibliography:\n" + bibliography_str)
    resp = model.generate_content([{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}], generation_config={"temperature": 0.4, "top_p": 0.9})
    return getattr(resp, "text", "(No answer produced)")

def analyze_intent(user_input: str, gemini_key: str, model_name: str = "gemini-2.5-flash") -> str:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)
    sys_prompt = textwrap.dedent("""
    You are an intent classifier for a research assistant tool. The user input could be:
    1) Asking for further refinement of the report.
    2) Asking a specific question about the report.
    3) Expressing satisfaction and wanting to finish.

    Classify the intent into exactly one of these labels: refine, ask, accept.
    Respond only with the label: refine, ask, or accept.
    """)
    user_prompt = f"User input: {user_input}"
    resp = model.generate_content([{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}], generation_config={"temperature": 0.0, "top_p": 0.9})
    intent = getattr(resp, "text", "ask").strip().lower()
    if intent not in ["refine", "ask", "accept"]:
        return "ask"
    return intent

def generate_query_from_input(user_input: str, gemini_key: str, model_name: str = "gemini-2.5-flash") -> str:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)
    sys_prompt = textwrap.dedent("""
    Convert the user's feedback into a concise, plain-text search query suitable for finding academic papers.
    The output should be a single line of text with no markdown, explanation, or extra details.
    """)
    user_prompt = f"User feedback: {user_input}"
    resp = model.generate_content([{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}], generation_config={"temperature": 0.0, "top_p": 0.9})
    return getattr(resp, "text", "").strip()

def generate_question_from_input(user_input: str, gemini_key: str, model_name: str = "gemini-2.5-flash") -> str:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)
    sys_prompt = textwrap.dedent("""
    Convert the user's statement into a precise research question.
    Output only the question in plain text, without markdown, explanation, or extra context.
    """)
    user_prompt = f"User statement: {user_input}"
    resp = model.generate_content([{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}], generation_config={"temperature": 0.0, "top_p": 0.9})
    return getattr(resp, "text", "").strip()

# -----------------------
# I/O helpers to provide downloads and saving to Mongo
# -----------------------
def save_outputs_db(chat_id: str, papers: List[Paper], report_md: str, out_base: Optional[str] = None) -> Optional[str]:
    papers_list = [p.to_row() for p in papers]
    update_chat_report_and_papers(chat_id, report_md or "", papers_list, meta={"saved_at": time.time()})
    return None

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Deep Research Agent", layout="wide")
st.markdown(
    """
        <style>
            .sidebar-chat {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 6px 8px;
                    margin: 4px 0;
                    border-radius: 6px;
                    background: #f9f9f9;
                }
        </style>
    """, 
    unsafe_allow_html=True)

st.title("🔬 Deep Research Agent")

# session login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Authentication
if not st.session_state.logged_in:
    st.header("Login / Signup")
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    with tab1:
        uname = st.text_input("Username", key="signin_username")
        pwd = st.text_input("Password", type="password", key="signin_password")
        if st.button("Sign In"):
            if authenticate_user(uname, pwd):
                st.session_state.logged_in = True
                st.session_state.username = uname
                user_chats = get_user_chats(uname)
                st.session_state.active_chat_id = user_chats[0]["_id"] if user_chats else create_chat(uname) ["_id"]
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        uname = st.text_input("New Username", key="signup_username")
        pwd1 = st.text_input("Password", type="password", key="signup_password1")
        pwd2 = st.text_input("Re-enter Password", type="password", key="signup_password2")
        if st.button("Sign Up"):
            if pwd1 != pwd2:
                st.error("Passwords do not match")
            else:
                if create_user(uname, pwd1):
                    st.success("Account created! Please sign in.")
                else:
                    st.error("Username exists")
    st.stop()

username = st.session_state.username

# Sidebar: chats + create
with st.sidebar:
    cols = st.columns([8,1])
    with cols[0]:
        st.markdown(f"#### **Hello,** {username}")
    with cols[1]:
        if st.button("➜]"):
            st.session_state.logged_in = False
            st.rerun()
    st.markdown("### 💬 Research Chats")
    if st.button("✚ Create New Chat", use_container_width=True):
        new_chat = create_chat(username)
        st.session_state.active_chat_id = new_chat["_id"]
        st.rerun()

    chats = get_user_chats(username)
    for chat in chats:
        cols = st.columns([8,1])
        with cols[0]:
            if st.button(chat.get("title") or "Untitled", key=f"chat_{chat['_id']}"):
                st.session_state.active_chat_id = chat["_id"]
                st.rerun()
        with cols[1]:
            if st.button("✖", key=f"del_{chat['_id']}"):
                chats_col.delete_one({"_id": chat["_id"]})
                if st.session_state.active_chat_id == chat["_id"]:
                    st.session_state.active_chat_id = None
                st.rerun()

# Main area (research-only)
if st.session_state.active_chat_id:
    chat = get_chat(st.session_state.active_chat_id)
    st.subheader(f"📝 {chat.get('title').capitalize() or 'Untitled'}")

    st.markdown("Use the panel below to run a research report, refine it, ask questions, or accept the report.")

    # Settings panel (open by default). Removed embed/model inputs (defaults used).
    with st.expander("**Research Settings**", expanded=True):
        topic = st.text_input("Research Topic", value=chat.get("meta", {}).get("last_topic", ""))
        max_papers = st.number_input("Max papers to fetch", min_value=10, max_value=500, value=chat.get("meta", {}).get("max_papers", 50))
        top_k = st.number_input("Top papers for context (Number of top papers based on which the report will be generated)", min_value=3, max_value=100, value=chat.get("meta", {}).get("top_k", 10))
        w_rel = st.number_input("Weight - Relevance", min_value=0.0, max_value=1.0, value=chat.get("meta", {}).get("w_rel", 0.4))
        w_cit = st.number_input("Weight - Citations", min_value=0.0, max_value=1.0, value=chat.get("meta", {}).get("w_cit", 0.25))
        w_rec = st.number_input("Weight - Recency", min_value=0.0, max_value=1.0, value=chat.get("meta", {}).get("w_rec", 0.35))
        st.caption("Note: Weights should ideally sum to 1.0, but the system will normalize them if they don't.")
        # embedding & generation models intentionally removed (defaults used)

    # Initialize session_state holders for ranked papers, QA history and last topic if missing.
    if "ranked_papers" not in st.session_state:
        st.session_state.ranked_papers = None
    if "last_fetch_topic" not in st.session_state:
        st.session_state.last_fetch_topic = None
    if "feedback_value" not in st.session_state:
        st.session_state.feedback_value = ""
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # If a chat is active, load any saved ranked_papers and qa_history into session_state
    if st.session_state.get("active_chat_id"):
        chat_doc = get_chat(st.session_state.active_chat_id)
        if chat_doc:
            # load ranked papers (if present) into session_state
            if chat_doc.get("ranked_papers"):
                st.session_state.ranked_papers = chat_doc.get("ranked_papers", None)
                st.session_state.last_fetch_topic = chat_doc.get("meta", {}).get("last_topic", st.session_state.last_fetch_topic)
            # load QA history
            if chat_doc.get("qa_history") is not None:
                st.session_state.qa_history = chat_doc.get("qa_history", [])

    # Button: fetch+rank (shows ranking table)
    if st.button("▶️ Run Research (Fetch & Rank)"):
        if not topic:
            st.error("Please provide a topic.")
        else:
            api_keys = {
                "SEMANTIC_SCHOLAR_KEY": SEMANTIC_SCHOLAR_KEY,
                "SERPAPI_KEY": SERPAPI_KEY,
                "IEEE_KEY": IEEE_API_KEY
            }
            with st.spinner("Fetching papers..."):
                collected = fetch_papers(topic, max_papers, api_keys)
            if not collected:
                st.error("No papers found. Try broadening the query. If that doesn't work, then we might be having problem with our APIs.")
            else:
                collected = dedup_papers(collected)
                st.info(f"{len(collected)} papers after deduplication.")
                st.info("Scoring and ranking papers. This may take a bit (embedding calls).")
                ranked = score_and_rank(collected, topic, (w_rel, w_cit, w_rec), GEMINI_API_KEY)
                # store ranked papers in session_state (as list of serializable dicts)
                ranked_serialized = [p.to_row() for p in ranked]
                st.session_state.ranked_papers = ranked_serialized
                st.session_state.last_fetch_topic = topic

                # persist to DB so the user can return later and generate report from the ranking
                try:
                    update_chat_ranked_papers(chat["_id"], ranked_serialized, meta={"last_topic": topic, "max_papers": max_papers, "top_k": top_k, "w_rel": w_rel, "w_cit": w_cit, "w_rec": w_rec})
                    st.success("Ranking complete and persisted to chat. Inspect the table below and click 'Generate Report' when ready.")
                except Exception as e:
                    st.error(f"Failed to persist ranked papers to DB: {e}")
                    st.success("Ranking complete (in-session). Inspect the table below and click 'Generate Report' when ready.")

                st.rerun()

    # If we have ranked papers in session_state for current topic, show ranking table (numbered from 1)
    if st.session_state.ranked_papers and st.session_state.last_fetch_topic == topic:
        st.markdown("### 🔢 Ranked Papers (Top results)")
        ranked_preview = st.session_state.ranked_papers
        # normalize citation_count values (coerce to int where possible)
        for p in ranked_preview:
            val = p.get("citation_count")
            try:
                p["citation_count"] = int(val) if val is not None else 0
            except Exception:
                p["citation_count"] = 0
        import pandas as pd
        df = pd.DataFrame(ranked_preview)
        display_cols = ["title", "year", "citation_count", "similarity", "score", "source"]
        # keep only existing columns
        display_cols = [c for c in display_cols if c in df.columns]
        display_df = df[display_cols].copy()
        display_df = display_df.rename(columns={"title": "Title", "year": "Year", "citation_count": "Cites", "similarity": "Relevance", "score": "Overall Score", "source": "Source"})
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(display_df.head(100))

        # Generate report button (after inspection)
        if st.button("📝 Generate Report"):
            # reconstruct Paper objects (prefer session_state, fall back to DB)
            ranked_source = st.session_state.get("ranked_papers")
            if not ranked_source:
                # fallback to DB-stored ranked papers
                chat_refreshed = get_chat(chat["_id"])
                ranked_source = chat_refreshed.get("ranked_papers", [])

            ranked_objs = []
            for d in (ranked_source or []):
                # ensure citation_count integer or None
                c = d.get("citation_count")
                try:
                    c = int(c) if c is not None else None
                except Exception:
                    c = None
                d["citation_count"] = c
                ranked_objs.append(Paper(**d))
            with st.spinner("Generating Report..."):
                report_md = generate_report(topic, ranked_objs, top_k, GEMINI_API_KEY)
            # save to local and DB
            md_path = save_outputs_db(chat["_id"], ranked_objs, report_md, out_base=f"report_{chat['_id']}")
            st.success("Report generated and saved to chat.")
            # update chat title/meta
            if chat.get("title", "").startswith(""):
                new_title = f"{topic[:50]}"
                chats_col.update_one({"_id": chat["_id"]}, {"$set": {"title": new_title}})
            chats_col.update_one({"_id": chat["_id"]}, {"$set": {"meta.last_topic": topic, "meta.max_papers": max_papers, "meta.top_k": top_k, "meta.w_rel": w_rel, "meta.w_cit": w_cit, "meta.w_rec": w_rec}})
            # refresh chat from DB to include report + papers
            st.rerun()

    # Show existing saved report if present in DB
    chat = get_chat(st.session_state.active_chat_id)  # reload
    if chat.get("report_md"):
        st.markdown("### 📄 Latest Report")
        pdf_bytes = markdown_to_pdf_bytes(chat["report_md"])
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        st.download_button(
            label="Download Report (as PDF)",
            data=pdf_bytes,
            file_name=f"{chat['title'].replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
        with st.expander("Show report (markdown)"):
            st.code(chat["report_md"][:200000])

        # Refinement & Q&A
        st.markdown("### 🔁 Refinement & QnA")

        # Ensure storage for feedback and QA history
        if "feedback_value" not in st.session_state:
            st.session_state.feedback_value = ""
        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []  # list of {"q": ..., "a": ...}

        # Render QA history first (so user sees past Q&A)
        if st.session_state.qa_history:
            for i, pair in enumerate(st.session_state.qa_history, 1):
                st.markdown(f"**Q{i}:** **{pair.get('q')}**")
                st.markdown(f"**A{i}:** {pair.get('a')}")
                st.markdown("---")

        # Show text area (widget key: 'feedback_widget'), value bound to feedback_value
        user_feedback = st.text_area(
            "Enter feedback to refine the report, ask a question, or say 'accept' to finish.",
            key="feedback_widget",
            value=st.session_state.get("feedback_value", ""),
            height=140
        )
        # Keep stored copy in feedback_value
        if user_feedback != st.session_state.get("feedback_value", ""):
            st.session_state.feedback_value = user_feedback

        col1, col2 = st.columns(2)
        if col1.button("Classify Intent"):
            if not st.session_state.feedback_value.strip():
                st.warning("Type something first.")
            else:
                intent = analyze_intent(st.session_state.feedback_value, GEMINI_API_KEY)
                st.info(f"Intent detected: **{intent}**")

        if col2.button("Process Input"):
            if not st.session_state.feedback_value.strip():
                st.warning("Type something first.")
            else:
                user_text = st.session_state.feedback_value
                intent = analyze_intent(user_text, GEMINI_API_KEY)
                
                if intent == "accept":
                    st.success("Marked as accepted. Report finalized.")
                    # clear stored feedback and leave QA history intact
                    st.session_state.feedback_value = ""
                    st.rerun()

                elif intent == "ask":
                    # convert to precise question and get answer
                    question = generate_question_from_input(user_text, GEMINI_API_KEY)
                    st.info(f"Generated question: {question}")

                    # Show a placeholder for the answer and display a loader below the question while Gemini answers
                    # First, show the question as a permanent element
                    st.markdown("**Question:**")
                    st.markdown(f"> {question}")

                    # Then show a spinner (loader) below it while fetching the answer
                    answer_placeholder = st.empty()
                    with st.spinner("Generating answer..."):
                        papers_objs = [Paper(**p) for p in (chat.get("papers") or [])]
                        answer = answer_question(question, chat.get("report_md", ""), papers_objs, chat.get("meta", {}).get("top_k", 10), GEMINI_API_KEY)
                        # populate the placeholder with the answer once ready
                        answer_placeholder.markdown("**Answer:**\n\n" + answer)

                    # append to QA history in session_state
                    st.session_state.qa_history.append({"q": question, "a": answer})
                    # persist to DB
                    try:
                        append_chat_qa(chat["_id"], question, answer)
                    except Exception as e:
                        st.error(f"Failed to persist QA to DB: {e}")

                    # clear stored feedback and rerun to clear the widget while keeping QA history visible
                    st.session_state.feedback_value = ""
                    st.rerun()

                elif intent == "refine":
                    refinement_prompt = generate_query_from_input(user_text, GEMINI_API_KEY)
                    st.info(f"Refining results with: '{refinement_prompt}' ...")
                    api_keys = {"SEMANTIC_SCHOLAR_KEY": SEMANTIC_SCHOLAR_KEY, "SERPAPI_KEY": SERPAPI_KEY, "IEEE_KEY": IEEE_API_KEY}

                    # Fetch new papers based on refinement prompt
                    with st.spinner("Fetching additional papers for refinement..."):
                        new_papers = fetch_papers(refinement_prompt, max(10, int(chat.get("meta", {}).get("max_papers", 80) // 2)), api_keys)
                        if not new_papers:
                            st.error("No new papers found for refinement query.")
                            st.stop()
                        
                    existing_papers = [Paper(**p) for p in (chat.get("papers") or [])]
                    new_papers = exclude_papers(new_papers, existing_papers[:int(chat.get("meta", {}).get("top_k", 25))])
                    merged = dedup_papers(existing_papers + new_papers)
                        
                    weights = (chat.get("meta", {}).get("w_rel", 0.4), chat.get("meta", {}).get("w_cit", 0.25), chat.get("meta", {}).get("w_rec", 0.35))
                    st.info("Re-ranking the papers based on refinement prompt...")
                    reranked = score_and_rank(merged, refinement_prompt, weights, GEMINI_API_KEY)
                    reranked_serialized = [p.to_row() for p in reranked]

                    # Persist reranked results
                    update_chat_ranked_papers(chat["_id"], reranked_serialized, meta={"last_topic": refinement_prompt})
                    st.session_state.ranked_papers = reranked_serialized
                    st.session_state.last_fetch_topic = refinement_prompt

                    # Log in QA history that refinement occurred
                    refine_note = f"🧠 Paper Refinement applied: **{refinement_prompt}**"
                    st.session_state.qa_history.append({"q": refine_note, "a": "Ranking updated, which you can check from the updated table at the top. Also, you can now generate the new report if you want from the button below the table."})
                    append_chat_qa(chat["_id"], refine_note, "Ranking updated, which you can check from the updated table at the top. Also, you can now generate the new report if you want from the button below the table.")

                    st.success("Refinement complete. Review the new ranked table below, then click 'Generate Updated Report' when ready.")

                    st.session_state.feedback_value = ""
                    st.rerun()

    else:
        st.info("No report generated yet for this research chat. Please run the research pipeline above.")

else:
    st.info("Select or create a research chat from the sidebar.")
