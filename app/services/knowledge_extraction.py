"""
Stage 3: Knowledge Extraction.

Builds the structured KnowledgeGraph (objectives, prerequisites, concepts,
formulae, examples, misconceptions) that every downstream stage -- the
teaching planner, content generator, and gap analysis -- reads from
instead of re-reading the raw document. Centralizing extraction here means
Stage 4 onward never has to reason about raw unstructured text again,
which is what keeps the later prompts short and reliable.

For documents longer than a single context-friendly chunk, this chunks and
merges rather than truncating, since silently dropping the back half of a
chapter would understate prerequisites and misconceptions that often show
up in worked examples near the end.
"""

from app.llm_client import generate_structured
from app.models.schemas import EducationalMetadata, KnowledgeGraph, ParsedDocument

SYSTEM_PROMPT = """You are an expert instructional designer extracting a \
structured knowledge representation from an educational document. Adapt your \
extraction to the document's domain (given in the metadata) -- for STEM \
content, extract formulae and precise definitions; for Humanities content, \
extract arguments, themes, and interpretive frameworks in place of formulae \
(leave related_formulae empty where none genuinely apply, do not invent \
them). Extract common student misconceptions specifically, since these \
directly feed a later lesson-gap-analysis stage and are one of the more \
undervalued fields here -- do not skip them."""

CHUNK_CHARS = 8000


def extract_knowledge(doc: ParsedDocument, metadata: EducationalMetadata) -> KnowledgeGraph:
    chunks = _chunk_text(doc.raw_text, CHUNK_CHARS)

    if len(chunks) == 1:
        return _extract_single(chunks[0], metadata)

    partials = [_extract_single(chunk, metadata) for chunk in chunks]
    return _merge_knowledge_graphs(partials)


def _extract_single(text_chunk: str, metadata: EducationalMetadata) -> KnowledgeGraph:
    user_prompt = (
        f"Subject: {metadata.subject}\nGrade: {metadata.grade}\n"
        f"Category: {metadata.category}\nDifficulty: {metadata.difficulty}\n\n"
        f"Document text:\n{text_chunk}"
    )
    print(">>> Entered Knowledge Extraction")
    print(f"Chunk length: {len(text_chunk)}")
    result = generate_structured(
    SYSTEM_PROMPT,
    user_prompt,
    KnowledgeGraph,
    max_tokens=30072,   
)
    print(">>> Knowledge Extraction Complete")
    return result


def _chunk_text(text: str, chunk_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    # Split on paragraph boundaries where possible to avoid cutting a
    # definition or worked example in half mid-sentence.
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_chars and current:
            chunks.append(current)
            current = para
        else:
            current += "\n" + para
    if current:
        chunks.append(current)
    return chunks


def _merge_knowledge_graphs(partials: list[KnowledgeGraph]) -> KnowledgeGraph:
    merged = KnowledgeGraph()
    seen_concepts: set[str] = set()

    for kg in partials:
        merged.learning_objectives.extend(kg.learning_objectives)
        merged.prerequisites.extend(kg.prerequisites)
        merged.keywords.extend(kg.keywords)
        merged.examples.extend(kg.examples)
        merged.applications.extend(kg.applications)
        merged.common_misconceptions.extend(kg.common_misconceptions)
        for concept in kg.concepts:
            if concept.name.lower() not in seen_concepts:
                seen_concepts.add(concept.name.lower())
                merged.concepts.append(concept)

    # Dedupe simple string lists while preserving order
    merged.learning_objectives = _dedupe(merged.learning_objectives)
    merged.prerequisites = _dedupe(merged.prerequisites)
    merged.keywords = _dedupe(merged.keywords)
    merged.common_misconceptions = _dedupe(merged.common_misconceptions)
    return merged


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
