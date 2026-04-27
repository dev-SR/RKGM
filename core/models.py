from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class GapType(str, Enum):
    TERMINOLOGY = "terminology"
    METHODOLOGY = "methodology"
    BENCHMARK = "benchmark"
    HISTORICAL = "historical"
    MATHEMATICAL = "mathematical"


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Layer(str, Enum):
    FOUNDATION = "foundation"
    DEVELOPMENT = "development"
    FRONTIER = "frontier"


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    year: int
    citation_count: int
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    level: int = 0  # 0=BA, 1/2/3 = reference depth
    layer: Layer = Layer.DEVELOPMENT
    trendscore: float = 0.0


@dataclass
class KnowledgeGap:
    gap_id: str
    concept: str
    gap_type: GapType
    difficulty: Difficulty
    domain: str
    why_needed: str
    layer_hint: Layer
    retrieval_query: str
    source_passage: str
    confidence: float


@dataclass
class CandidatePaper:
    paper: Paper
    gap_id: str
    relevance_score: float
    rationale: str
    pdf_available: bool = False


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    section: str
    text: str
    page: int = 0
    embedding: Optional[list] = None


@dataclass
class GapExplanation:
    gap_id: str
    concept: str
    explanation_text: str
    source_citations: list
    confidence: float
    is_abstract_only: bool = False
    order_position: int = 0
    dependency_note: str = ""


@dataclass
class PipelineState:
    # Phase A
    ba_text: str = ""
    ba_paper: Optional[Paper] = None
    reference_graph: Optional[object] = None
    all_papers: dict = field(default_factory=dict)
    gaps: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    # Phase B
    chunks: list = field(default_factory=list)
    chroma_collection: Optional[object] = None
    bm25_index: Optional[object] = None
    explanations: list = field(default_factory=list)
    ordered_gap_ids: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    final_document: str = ""
    errors: list = field(default_factory=list)
