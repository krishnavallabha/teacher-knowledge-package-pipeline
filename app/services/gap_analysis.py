"""
Stage 8: Learning Gap Analysis.

Turns the misconceptions already surfaced in Stage 3 into something
actionable: a diagnostic question to catch the misconception in the
moment, a severity rating, and a concrete remedial action. Runs once for
the whole chapter rather than per period, since a misconception (e.g.
"students confuse velocity and acceleration") is usually a chapter-level
pattern, not scoped to one period.
"""

from app.llm_client import generate_structured
from app.models.schemas import EducationalMetadata, KnowledgeGraph, LearningGap
from pydantic import BaseModel, Field


class _GapList(BaseModel):
    gaps: list[LearningGap] = Field(default_factory=list)


SYSTEM_PROMPT = """You are an expert in diagnostic assessment and \
misconception analysis. For each given misconception, write one short \
diagnostic question that would reveal whether a student holds that \
misconception, rate its severity (low/medium/high) based on how much it \
would block downstream understanding, and give one concrete remedial \
action a teacher can take in the next class."""


def analyze_gaps(knowledge: KnowledgeGraph, metadata: EducationalMetadata) -> list[LearningGap]:
    if not knowledge.common_misconceptions:
        return []

    user_prompt = (
        f"Subject: {metadata.subject}, Grade: {metadata.grade}\n"
        f"Misconceptions to analyze: {knowledge.common_misconceptions}\n"
    )
    result = generate_structured(SYSTEM_PROMPT, user_prompt, _GapList, max_tokens=3024)
    return result.gaps
