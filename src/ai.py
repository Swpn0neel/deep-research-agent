import textwrap
import numpy as np
from typing import List, Tuple
import google.generativeai as genai
from .models import Paper

def _setup_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai

def gemini_embed(embed_fn, text: str, task_type: str = "retrieval_query", model: str = "models/text-embedding-004") -> np.ndarray:
    res = embed_fn(model=model, content=text, task_type=task_type)
    vec = np.array(res["embedding"], dtype=np.float32)
    return vec

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

def generate_report(topic: str, papers: List[Paper], top_k: int, gemini_api_key: str, model_name: str = "models/gemini-3-flash-preview") -> str:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name)

    context_str, bibliography_str = build_context_chunks(papers, top_k)

    sys_prompt = textwrap.dedent(f"""
    You are an expert research analyst. Given a topic and a set of top-ranked papers (with abstracts), write a THOROUGH, DETAILED, and COMPREHENSIVE research report with the following structure:

    1) Executive Summary (300-500 words)
       - Provide a broad, insightful overview of the field, highlighting the most significant findings, trends, and challenges.
       - Summarize the main contributions of the top papers, referencing them with [#] citations.

    2) In-Depth Background & Core Concepts (400-600 words)
       - Explain all relevant background, terminology, and foundational concepts in detail.
       - Include historical context, key definitions, and major theoretical frameworks.
       - Use clear, accessible language for non-experts.

    3) Comparative Literature Synthesis (600-900 words)
       - Analyze and compare the top papers in depth, discussing methodologies, datasets, benchmarks, and results.
       - Identify major research clusters, approaches, and their evolution over time.
       - Highlight consensus, controversies, and open debates, citing papers as [#].

    4) Critical Gap Analysis (300-500 words)
       - Identify and discuss methodological, data, evaluation, reproducibility, and scalability gaps in the literature.
       - Point out under-explored areas, limitations, and weaknesses, with specific paper references.

    5) Future Research Directions (300-500 words)
       - Propose prioritized, concrete, and measurable future research directions.
       - Suggest new methodologies, datasets, or evaluation strategies.
       - Discuss potential for interdisciplinary work and emerging trends.

    6) Risks, Ethics, and Limitations (200-400 words)
       - Analyze ethical, societal, and practical risks associated with the research area.
       - Discuss limitations of current approaches and possible negative impacts.

    7) Practical Applications and Tooling Landscape (200-400 words)
       - Survey real-world applications, tools, and systems based on the reviewed research.
       - Highlight industry adoption, open-source projects, and commercial products.

    8) Conclusion (150-300 words)
       - Synthesize the main insights, reiterate the importance of the topic, and summarize key takeaways.

    Rules:
    - Use clear section headings and subheadings.
    - Use inline numeric citations like [1], [2] that map to the provided bibliography.
    - If evidence is weak or missing, explicitly state so.
    - Be precise, avoid hand-waving, and support all claims with references.
    - Aim for a total length of 3000-5000 words, unless context is sparse.
    - Ensure the report is self-contained and highly informative for both experts and newcomers.
    """
    )

    user_prompt = f"Topic: {topic}\n\nTop Papers Context (ranked):\n" + context_str + "\n\nBibliography (use these citation indices):\n" + bibliography_str

    resp = model.generate_content(
        [
            {"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]},
        ],
        safety_settings=None,
        generation_config={
            "temperature": 0.6,
            "top_p": 0.9,
        }
    )
    try:
        md = resp.text
    except Exception as e:
        raise Exception(f"Gemini generation failed: {e}")

    refs_lines = []
    for i, p in enumerate(papers[:top_k], 1):
        title = (p.title or "Untitled").strip()
        url = p.url or p.pdf_url or (f"https://doi.org/{p.doi}" if p.doi else None)
        if url:
            title_md = f"[{title}]({url})"
        else:
            title_md = title
        year = str(p.year) if p.year else "n.d."
        line = f"{i}. {title_md} — ({year})"
        if p.url or p.pdf_url:
            link = p.url or p.pdf_url
            line += f"\n**Link**: {link}"
        refs_lines.append(line)

    references_md = "## References\n\n" + "\n\n".join(refs_lines) + "\n"
    final_md = md.rstrip() + "\n\n" + references_md
    return final_md

def answer_question(question: str, report_md: str, papers: List[Paper], top_k: int, gemini_api_key: str, model_name: str = "models/gemini-3-flash-preview") -> str:
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

def analyze_intent(user_input: str, gemini_key: str, model_name: str = "models/gemini-3.1-flash-lite-preview") -> str:
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

def generate_query_from_input(user_input: str, gemini_key: str, model_name: str = "models/gemini-3.1-flash-lite-preview") -> str:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)
    sys_prompt = textwrap.dedent("""
    Convert the user's feedback into a concise, plain-text search query suitable for finding academic papers.
    The output should be a single line of text with no markdown, explanation, or extra details.
    """)
    user_prompt = f"User feedback: {user_input}"
    resp = model.generate_content([{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}], generation_config={"temperature": 0.0, "top_p": 0.9})
    return getattr(resp, "text", "").strip()

def generate_question_from_input(user_input: str, gemini_key: str, model_name: str = "models/gemini-3.1-flash-lite-preview") -> str:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)
    sys_prompt = textwrap.dedent("""
    Convert the user's statement into a precise research question.
    Output only the question in plain text, without markdown, explanation, or extra context.
    """)
    user_prompt = f"User statement: {user_input}"
    resp = model.generate_content([{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}], generation_config={"temperature": 0.0, "top_p": 0.9})
    return getattr(resp, "text", "").strip()

query_cache = {}
def enrich_research_query(query: str, gemini_key: str, model_name: str = "models/gemini-3.1-flash-lite-preview") -> str:
    if query in query_cache:
        return query_cache[query]
    
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(model_name)

    sys_prompt = textwrap.dedent("""
        You improve academic literature search queries.

        Rules:
        - Output must be a concise academic search query, not a sentence.
        - Avoid filler words like: investigating, exploring, study of, analysis of, research on.
        - Keep only meaningful technical terms.
        - Preserve the core research topic.
        - Expand with some relevant technical terms if useful.
        - Maximum 50 words.
        - No punctuation except hyphens if needed.

        Before producing the final query, ensure the result reads as a coherent research question or search phrase rather than a list of keywords.

        Output ONLY the rewritten research query.
    """)

    user_prompt = f"Original research query: {query}"
    resp = model.generate_content(
        [{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}],
        generation_config={"temperature": 0.0, "top_p": 1.0}
    )

    enriched = getattr(resp, "text", query).strip()
    query_cache[query] = enriched
    return enriched
