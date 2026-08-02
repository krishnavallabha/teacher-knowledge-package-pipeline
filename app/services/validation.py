"""
Stage 9: Validation.

Deliberately pure Python, no LLM call. Schema adherence is already
guaranteed by generate_structured's Pydantic validation at each stage, so
this stage focuses on the checks that need to look ACROSS stages: did every
learning objective actually get covered by some period's content, does
every period referenced in activities/assessments actually exist in the
teaching plan, are there any periods with no assessment at all. Running
this as code rather than another LLM call also means it's fast, free, and
deterministic, which is exactly what a validation gate should be.
"""
import re

from rapidfuzz import fuzz
from app.models.schemas import TeacherKnowledgePackage, ValidationIssue, ValidationReport

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    stop_words = {
        "the", "a", "an", "to", "of", "in",
        "simple", "simply", "what", "is",
        "describe", "explain", "define",
        "identify", "state"
    }

    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


def _matches(a: str, b: str) -> bool:
    return (
        fuzz.token_sort_ratio(
            _normalize(a),
            _normalize(b)
        ) >= 85
    )
def validate_package(tkp: TeacherKnowledgePackage) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if not tkp.metadata:
        issues.append(_error("classification", "Missing educational metadata"))
    if not tkp.knowledge or not tkp.knowledge.concepts:
        issues.append(_error("knowledge-extraction", "No concepts extracted"))
    if not tkp.teaching_plan or not tkp.teaching_plan.periods:
        issues.append(_error("teaching-planner", "No teaching plan periods generated"))
        return ValidationReport(passed=False, issues=issues)

    plan_period_numbers = {p.period_number for p in tkp.teaching_plan.periods}

    # Every objective should be traceable to at least one period covering
    # a concept it names or being present in that period's own objectives.
    if tkp.knowledge:
        covered_objectives = [
    obj
    for p in tkp.teaching_plan.periods
    for obj in p.learning_objectives
]

        for objective in tkp.knowledge.learning_objectives:
            if not any(
        _matches(objective, planner_obj)
        for planner_obj in covered_objectives
    ):
                issues.append(
                    ValidationIssue(
                        stage="teaching-planner",
                        severity="warning",
                        message=f"Learning objective not explicitly assigned to any period: {objective}",
                    )
                )

    content_periods = {c.period_number for c in tkp.period_contents}
    missing_content = plan_period_numbers - content_periods
    for pn in sorted(missing_content):
        issues.append(_error("content-generation", f"Period {pn} has no generated classroom content"))

    assessment_periods = {a.period_number for a in tkp.assessments}
    missing_assessment = plan_period_numbers - assessment_periods
    for pn in sorted(missing_assessment):
        issues.append(_error("assessment-generation", f"Period {pn} has no assessment"))

    for period_content in tkp.period_contents:
        if period_content.period_number not in plan_period_numbers:
            issues.append(
                _error(
                    "content-generation",
                    f"Content references period {period_content.period_number} not in teaching plan",
                )
            )

    for assessment in tkp.assessments:
        for q in assessment.questions:
            if q.question_type == "mcq" and (not q.options or len(q.options) != 4):
                issues.append(
                    ValidationIssue(
                        stage="assessment-generation",
                        severity="warning",
                        message=f"MCQ in period {assessment.period_number} does not have exactly 4 options",
                    )
                )
            if not q.correct_answer.strip():
                issues.append(
                    _error(
                        "assessment-generation",
                        f"Question in period {assessment.period_number} missing a correct answer",
                    )
                )

    passed = not any(i.severity == "error" for i in issues)
    return ValidationReport(passed=passed, issues=issues)


def _error(stage: str, message: str) -> ValidationIssue:
    return ValidationIssue(stage=stage, severity="error", message=message)
