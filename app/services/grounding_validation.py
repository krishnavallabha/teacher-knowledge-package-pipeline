"""
Stage 9b: Grounding Validation.

This is distinct from Stage 9's structural validation (validation.py),
which only checks internal consistency between pipeline stages -- it never
looks back at the original document at all, so it would happily pass a
lesson plan that is perfectly self-consistent but confidently teaches a
fact the source material never stated.

Per the client clarification, hallucination is specifically defined as
subject-matter content not traceable to the primary source document.
Supplementary pedagogy -- analogies, activities, classroom management
suggestions, assessment framing -- is explicitly allowed to draw on
outside knowledge, since none of that is "content" in the graded sense.
This stage therefore only checks the fields that carry factual/conceptual
claims (period content and its assessment) and does not flag mentor
moments or activity descriptions, since flagging an analogy as "not in the
source" would be flagging exactly the kind of pedagogical enrichment the
client asked for.

Matching approach: cosine similarity over TF-IDF vectors (via
retrieval.py's DocumentIndex) between a citation and its best-matching
source chunk, rather than a strict literal substring/n-gram check. This
replaces an earlier version of this file that required a citation to share
a 4-word run with the source text verbatim -- that bar was measurably too
strict in practice: a citation that correctly paraphrases source content
("objects at rest remain at rest unless a force acts") routinely shares no
4-word run with the source's own phrasing ("An object at rest stays at
rest... unless acted upon by a net external force") despite being fully
grounded. Cosine similarity over shared vocabulary catches this case
because it scores partial term overlap rather than requiring an exact
contiguous match. The threshold is a judgment call, not a proof of
correctness -- see MIN_SIMILARITY below and the honest limitations note in
docs/IMPLEMENTATION_GUIDE.md.
"""

from __future__ import annotations

from app.models.schemas import GroundingReport, ParsedDocument, TeacherKnowledgePackage, UngroundedClaim
from app.services.retrieval import DocumentIndex, build_index

# Below this cosine similarity, a citation is treated as unsupported by the
# retrieved source. Chosen empirically as a middle ground: high enough to
# reject a citation that shares little more than common words with the
# source, low enough to accept a reasonable paraphrase. Tune this per
# deployment if you see either too many false positives (raise it) or
# hallucinated content slipping through (lower it).
MIN_SIMILARITY = 0.18


def validate_grounding(tkp: TeacherKnowledgePackage, doc: ParsedDocument) -> GroundingReport:
    if not tkp.knowledge:
        return GroundingReport(passed=True)

    index = build_index(doc)
    all_claims: list[UngroundedClaim] = []

    for pc in tkp.period_contents:
        citations = pc.source_citations or []
        if not citations:
            all_claims.append(
                UngroundedClaim(
                    location=f"Period {pc.period_number} content",
                    claim="Missing source citations for generated content",
                    reason="No citation trace provided",
                )
            )
        for citation in citations:
            match = _check_citation(citation, index)
            if match is None:
                all_claims.append(
                    UngroundedClaim(
                        location=f"Period {pc.period_number} content",
                        claim=citation,
                        reason="Citation does not sufficiently match any retrieved source passage",
                    )
                )

    for assessment in tkp.assessments:
        for q in assessment.questions:
            citations = q.source_citations or []
            if not citations:
                all_claims.append(
                    UngroundedClaim(
                        location=f"Period {assessment.period_number} assessment",
                        claim=q.question_text,
                        reason="No citation trace provided",
                    )
                )
            for citation in citations:
                match = _check_citation(citation, index)
                if match is None:
                    all_claims.append(
                        UngroundedClaim(
                            location=f"Period {assessment.period_number} assessment",
                            claim=citation,
                            reason="Citation does not sufficiently match any retrieved source passage",
                        )
                    )

    return GroundingReport(passed=len(all_claims) == 0, ungrounded_claims=all_claims)


def _check_citation(citation: str, index: DocumentIndex):
    if not citation.strip():
        return None
    match = index.best_match(citation)
    if match is None:
        
        return None
    else:
     
     chunk, score = match
    chunk, score = match
    if score < MIN_SIMILARITY:
        return None
    return chunk, score
