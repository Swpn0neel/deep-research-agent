import requests
from urllib.parse import quote
from typing import List, Optional
from .models import Paper
from .config import NOW_YEAR

# Shared session for connection pooling
session = requests.Session()

def _get_json(url: str, params: dict = None, headers: dict = None, timeout: int = 30) -> dict:
    resp = session.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def _head(url: str, timeout: int = 8) -> bool:
    try:
        resp = session.head(url, timeout=timeout, allow_redirects=True)
        return resp.ok
    except requests.RequestException:
        return False

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
