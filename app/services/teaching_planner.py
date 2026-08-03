"""
Stage 4: Teaching Planner.

Splits the extracted knowledge into a sequence of class periods. This is
the pedagogical-flow stage: it decides which concepts go in which period,
in what order, and for how long, which every later content/activity/
assessment stage then generates material for period-by-period rather than
for the chapter as a whole. Getting sequencing wrong here (e.g. a concept
used before its prerequisite is taught) is a correctness bug just as much
as a code bug, so the prompt explicitly asks the model to respect the
prerequisite list.

Per the client clarification, neither the period count nor the period
length should be assumed fixed (no hardcoded "5 x 40 minutes") -- the
system should size both based on content volume, conceptual complexity,
and grade level, adapting naturally rather than forcing material into
predefined slots. A user-supplied time_constraint_minutes (from the
Stage 0 clarifying questions) is treated as a soft total-time budget the
plan should fit within, not a per-period duration -- the model still
decides how to divide that budget across periods of varying length.
"""

from app.llm_client import generate_structured
from app.models.schemas import EducationalMetadata, KnowledgeGraph, TeachingPlan, UserContext

SYSTEM_PROMPT = """You are an expert lesson planner. Convert a knowledge \
graph into a sequenced, multi-period teaching plan. Respect prerequisite \
ordering: a concept must not appear in an earlier period than the concepts \
it depends on. Each period should have a coherent, teachable scope -- do \
not just evenly split the concept list without regard to how they relate.

Do not assign more than 5-6 concepts to a single period, even for a dense \
chapter -- split into more periods instead of overloading one. A period \
whose content, activity, and assessment all have to cover too many concepts \
at once produces worse material and is more likely to hit generation limits \
than the same content spread across an extra period.

Decide BOTH the number of periods AND the duration of each period yourself, \
based on content volume, conceptual density, and grade level -- do not \
default to a fixed period count or a fixed duration like 40 minutes unless \
the content and constraints genuinely call for it. A short, simple topic \
for a younger grade may need only 1-2 short periods; a dense, multi-concept \
chapter for an older grade may need 5+ longer periods. Periods do not need \
to be the same length as each other.

Phrase each period's learning_objectives using Bloom's Taxonomy verbs at a \
level appropriate to that point in the sequence -- early periods introducing \
a concept can target Remember/Understand (define, describe, explain), while \
later periods building on that concept should target higher levels \
(Apply, Analyze, Evaluate, Create) rather than restating the same recall-level \
objective in different words. Not every objective needs to be high-level, but \
the plan as a whole should show progression, not a flat list of definitions."""


def build_teaching_plan(
    knowledge: KnowledgeGraph,
    metadata: EducationalMetadata,
    user_context: UserContext | None = None,
) -> TeachingPlan:
    concept_names = [c.name for c in knowledge.concepts]

    constraint_lines = []
    if user_context:
        if user_context.time_constraint_minutes:
            constraint_lines.append(
                f"Total instructional time available: {user_context.time_constraint_minutes} minutes "
                "(fit the full plan within this budget; you still decide how it's divided across periods)."
            )
        if user_context.teaching_objectives:
            constraint_lines.append(f"Teacher-stated objectives to prioritize: {user_context.teaching_objectives}")
        if user_context.teaching_style:
            constraint_lines.append(f"Preferred teaching style: {user_context.teaching_style}")
        if user_context.grade_override:
            constraint_lines.append(f"Confirmed grade level (overrides inferred grade): {user_context.grade_override}")

    user_prompt = (
        f"Subject: {metadata.subject}, Grade: {user_context.grade_override if user_context and user_context.grade_override else metadata.grade}\n"
        f"Difficulty: {metadata.difficulty}\n"
        + ("\n".join(constraint_lines) + "\n" if constraint_lines else "No additional teacher constraints provided -- size the plan purely from content.\n")
        + f"\nLearning objectives: {knowledge.learning_objectives}\n"
        f"Prerequisites: {knowledge.prerequisites}\n"
        f"Concepts (in no particular order): {concept_names}\n"
    )
    return generate_structured(SYSTEM_PROMPT, user_prompt, TeachingPlan, max_tokens=6000)
