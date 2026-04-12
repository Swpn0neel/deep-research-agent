import streamlit as st
from typing import List, Dict, Tuple
from .models import Paper
from .clients import SemanticScholarClient, ArxivClient, CrossrefClient, SerpAPIGoogleScholarClient, IEEEXploreClient
from .utils import _log1p_normalized, _recency_score, cosine
from .ai import _setup_gemini, gemini_embed

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

def score_and_rank(papers: List[Paper], topic: str, weights: Tuple[float, float, float], gemini_api_key: str, embed_model: str = "models/text-embedding-004", model_name: str = "models/gemini-3.1-flash-lite-preview") -> List[Paper]:
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
