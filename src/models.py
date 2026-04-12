from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

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
