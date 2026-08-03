"""Stage 5/6/7: Combined period generation.

This merges what used to be separate content, activity, and assessment
calls into a single model invocation per period. The goal is lower
latency, fewer round-trips, and less token overhead per period.

Grounding source: rather than sending a blind truncated excerpt of the
whole document (the old approach), this retrieves the chunks most relevant
to THIS period's specific concepts and objectives via retrieval.py, and
gives the model only those as source material. Two benefits: the model
gets a focused, on-topic context window instead of competing for
attention against unrelated sections of the chapter, and the citations it
writes are naturally anchored to material it was actually shown, rather
than material it half-remembers seeing 6000 characters up.
"""

from __future__ import annotations

from app.llm_client import generate_structured
from app.models.schemas import (
    EducationalMetadata,
    KnowledgeGraph,
    ParsedDocument,
    Period,
    PeriodGenerationBundle,
)
from app.services.retrieval import DocumentIndex, build_index, format_chunks_for_prompt

SYSTEM_PROMPT = """You are an expert teacher designing a single lesson period.
Return ONLY one JSON object with exactly these top-level keys:
content, activity, assessment.

Each nested object must match the schema for that object, including
period_number and source_citations. source_citations should be a short list
of brief, human-readable references to the RETRIEVED SOURCE PASSAGES below
(for example: page numbers, heading names, or short quoted phrases) that
justify what you wrote. Ground your content specifically in the retrieved
passages provided -- they were selected as the most relevant material for
this period's concepts, not the whole document, so use them as your primary
source rather than general knowledge about the subject.

Write checkpoint_questions and assessment questions to span a RANGE of
Bloom's Taxonomy levels rather than clustering at one level -- include at
least one question testing straightforward recall (Remember/Understand) and
at least one requiring the student to apply, analyze, or evaluate the
concept, not just restate a definition. Do not write every question as pure
recall.

Rules:
- Do not add any outer wrapper key.
- Do not invent citations that are not supported by the retrieved passages.
- Keep the activity practical and age-appropriate.
- Keep the assessment aligned to the period objectives and source content.
- The content should be concise.

Limits:
- teacher_script: maximum 180 words
- blackboard_notes: maximum 8 bullet points
- classroom_activities: maximum 2 activities
- checkpoint_questions: maximum 3
- exit_ticket: maximum 2 questions
- homework: maximum 2 questions
- mentor_moment: maximum 80 words

Do not repeat explanations already given elsewhere.
- The assessment should be a small but useful mix of question types.
"""

# Document indexes are expensive to build (a TF-IDF fit over all chunks)
# and every period for the same document needs the same index -- cached
# per ParsedDocument identity rather than rebuilt once per period call.
_index_cache: dict[int, DocumentIndex] = {}


def _get_or_build_index(doc: ParsedDocument) -> DocumentIndex:
    key = id(doc)
    if key not in _index_cache:
        _index_cache[key] = build_index(doc)
    return _index_cache[key]


def generate_period_bundle(
    period: Period,
    knowledge: KnowledgeGraph,
    metadata: EducationalMetadata,
    doc: ParsedDocument,
    review_notes: str | None = None,
) -> PeriodGenerationBundle:
    index = _get_or_build_index(doc)
    retrieval_query = " ".join(
        [period.title, *period.concepts_covered, *period.learning_objectives]
    )
    retrieved = index.retrieve(retrieval_query, top_k=3, min_score=0.05)
    source_passages = format_chunks_for_prompt(retrieved)
    relevant_concepts = _relevant_concepts(period, knowledge)
    # Cap what a single period is asked to cover. On a content-dense
    # chapter, the planner can still assign more concepts to one period
    # than a single content+activity+assessment call can write well within
    # any fixed token budget -- capping here bounds the request itself
    # rather than just hoping a bigger max_tokens absorbs an unbounded one.
    if len(relevant_concepts) > 6:
        relevant_concepts = relevant_concepts[:6]

    user_prompt = (
        f"Subject: {metadata.subject}\n"
        f"Grade: {metadata.grade}\n"
        f"Difficulty: {metadata.difficulty}\n"
        f"Topic: {metadata.topic}\n"
        f"Period number: {period.period_number}\n"
        f"Period title: {period.title}\n"
        f"Period duration: {period.duration_minutes} minutes\n"
        f"Period learning objectives: {period.learning_objectives}\n"
        f"Period concepts: {period.concepts_covered}\n"
        f"Relevant concepts: {[c.name for c in relevant_concepts]}\n\n"
        f"RETRIEVED SOURCE PASSAGES (most relevant to this period, ranked by relevance):\n{source_passages}\n\n"
        f"REVIEWER NOTES (if any):\n{review_notes or 'None'}\n\n"
        "Create the content, activity, and assessment together for this one period. "
        "Each nested object should include source_citations that point back to the "
        "retrieved passages above."
    )

    bundle = generate_structured(SYSTEM_PROMPT, user_prompt, PeriodGenerationBundle, max_tokens=8192)

    bundle.content.period_number = period.period_number
    bundle.activity.period_number = period.period_number
    bundle.assessment.period_number = period.period_number
    return bundle


def _relevant_concepts(period: Period, knowledge: KnowledgeGraph):
    period_concepts = {c.lower() for c in period.concepts_covered}
    return [c for c in knowledge.concepts if c.name.lower() in period_concepts]
