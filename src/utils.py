import math
import bcrypt
import numpy as np
from typing import Optional, Tuple
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from io import BytesIO
import markdown as md_lib
from bs4 import BeautifulSoup
from .config import NOW_YEAR

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)

def _log1p_normalized(x: Optional[int]) -> float:
    if not x or x <= 0:
        return 0.0
    return min(1.0, math.log1p(x) / math.log(1 + 1000))

def _recency_score(year: Optional[int]) -> float:
    if not year:
        return 0.0
    age = max(0, NOW_YEAR - int(year))
    return max(0.0, 1.0 - (age / 12.0))

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def markdown_to_pdf_bytes(markdown_text: str, title: Optional[str] = None) -> bytes:
    """Convert markdown text to styled PDF with bullet list support."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title=(title or "Report"))
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

    for elem in soup.body or soup:
        if getattr(elem, "name", None):
            handle_element(elem)

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
