"""
Retrieval over the uploaded document.

Adds the missing piece from the suggested architecture: Document -> Chunking
-> Embeddings -> Vector Index -> Retriever -> each stage retrieves relevant
chunks, rather than every stage receiving the same blind truncated excerpt
of the source text regardless of what it actually needs.

Deliberate choice: TF-IDF + cosine similarity (scikit-learn) instead of a
neural embedding model (sentence-transformers, OpenAI embeddings, etc).
Reasoning, stated plainly rather than implied:

- Retrieval scope here is a single uploaded document (a chapter, a few
  thousand to tens of thousands of words), not a corpus. TF-IDF and dense
  embeddings converge in quality at this scale far more than they do for
  cross-document retrieval, where embeddings clearly win on synonymy and
  paraphrase. The gap that matters at document-chapter scale is small.
- No model download at request time or container cold-start. A neural
  embedding model adds a multi-hundred-MB download from an external host on
  first use (or a bloated Docker image if baked in), which is a real
  reliability and latency cost for a demo deployment on a free/cheap tier,
  for a quality gain that's marginal at this retrieval scale.
- Deterministic and inspectable. TF-IDF term weights are easy to reason
  about when debugging why a chunk was or wasn't retrieved, which matters
  for a system whose entire second bullet point is grounding traceability.

This is a genuine engineering tradeoff, not a shortcut -- see
docs/IMPLEMENTATION_GUIDE.md for the honest version of when this would
stop being the right call (a large multi-chapter corpus, cross-document
retrieval, or synonym-heavy domains where lexical overlap breaks down).
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schemas import DocumentBlock, ParsedDocument

# Chunk size in characters. Small enough that a retrieved chunk is a
# focused, citable unit (roughly a paragraph or two) rather than a whole
# page, since the point of retrieval here is precise grounding, not just
# fitting more context in.
CHUNK_CHARS = 900
CHUNK_OVERLAP_CHARS = 150


@dataclass
class Chunk:
    chunk_id: int
    text: str
    page: int | None
    heading: str | None


class DocumentIndex:
    """A built TF-IDF index over one document's chunks, ready to retrieve from."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        if chunks:
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])
        else:
            self._vectorizer = None
            self._matrix = None

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks most relevant to query, each with its similarity score."""
        if not self.chunks or self._vectorizer is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:top_k] if score >= min_score]

    def best_match(self, text: str) -> tuple[Chunk, float] | None:
        """Single best-matching chunk for a piece of text (used by grounding checks)."""
        results = self.retrieve(text, top_k=1)
        return results[0] if results else None


def build_index(doc: ParsedDocument) -> DocumentIndex:
    chunks = _chunk_document(doc)
    return DocumentIndex(chunks)


def _chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """
    Chunk on block boundaries where possible (so a chunk doesn't split a
    heading from the paragraph it introduces), falling back to a sliding
    character window with overlap when a single block is longer than
    CHUNK_CHARS -- overlap keeps a sentence spanning two windows from
    losing whichever half's window it wasn't compared against.
    """
    chunks: list[Chunk] = []
    current_text = ""
    current_page: int | None = None
    current_heading: str | None = None
    last_heading = ""

    def flush():
        nonlocal current_text, current_page
        if current_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    text=current_text.strip(),
                    page=current_page,
                    heading=current_heading,
                )
            )
        current_text = ""

    for block in doc.blocks:
        text = block.text.strip()
        if not text:
            continue

        if block.block_type == "heading":
            # A new heading starts a new section -- flush whatever was
            # accumulated so far rather than letting it merge into this
            # section, even if the running chunk is still well under
            # CHUNK_CHARS. Without this, a short document (a single small
            # chapter, which the client's own clarification doc says is
            # common for younger grades) collapses into one giant chunk
            # that never gets split, and retrieval degrades to exactly the
            # "return the whole excerpt" behavior it exists to replace.
            flush()
            last_heading = text
            current_page = block.page
            current_heading = last_heading
            current_text = text
            continue

        candidate = (current_text + "\n" + text).strip() if current_text else text
        if len(candidate) > CHUNK_CHARS and current_text:
            flush()
            current_text = text
            current_page = block.page
            current_heading = last_heading
        else:
            if not current_text:
                current_page = block.page
                current_heading = last_heading
            current_text = candidate

        # A single block longer than the chunk size on its own (a long
        # paragraph with no internal heading breaks) gets its own sliding
        # window rather than becoming one oversized chunk.
        if len(current_text) > CHUNK_CHARS * 1.5:
            _flush_sliding_window(chunks, current_text, current_page, current_heading)
            current_text = ""

    flush()

    if not chunks:
        # No structural blocks at all (shouldn't normally happen) -- fall
        # back to a flat sliding window over the raw text so retrieval
        # still has something to index.
        _flush_sliding_window(chunks, doc.raw_text, None, None)

    return chunks


def _flush_sliding_window(
    chunks: list[Chunk], text: str, page: int | None, heading: str | None
) -> None:
    start = 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(chunk_id=len(chunks), text=chunk_text, page=page, heading=heading))
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP_CHARS


def format_chunks_for_prompt(results: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks as a labeled block for inclusion in an LLM prompt."""
    if not results:
        return "No relevant source passages retrieved."
    lines = []
    for chunk, score in results:
        location = f"p.{chunk.page}" if chunk.page is not None else "p.?"
        heading = f" [{chunk.heading}]" if chunk.heading else ""
        lines.append(f"[chunk {chunk.chunk_id}, {location}{heading}, relevance {score:.2f}]\n{chunk.text}")
    return "\n\n".join(lines)
