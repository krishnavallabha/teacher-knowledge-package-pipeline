"""Grounding correction pass.

When grounding validation flags claims, this stage reruns the combined
period-generation call for the affected periods with the flagged claims
explicitly provided as reviewer notes.
"""

from __future__ import annotations

import re

from app.models.schemas import ParsedDocument, TeacherKnowledgePackage
from app.services.period_generation import generate_period_bundle

_PERIOD_RE = re.compile(r"Period\s+(\d+)")


def correct_ungrounded_content(
    tkp: TeacherKnowledgePackage,
    doc: ParsedDocument,
) -> TeacherKnowledgePackage:
    if not tkp.grounding or not tkp.grounding.ungrounded_claims or not tkp.teaching_plan:
        return tkp

    claims_by_period: dict[int, list[str]] = {}
    for claim in tkp.grounding.ungrounded_claims:
        match = _PERIOD_RE.search(claim.location)
        if not match:
            continue
        period_number = int(match.group(1))
        claims_by_period.setdefault(period_number, []).append(f"{claim.location}: {claim.claim} ({claim.reason})")

    if not claims_by_period:
        return tkp

    period_to_index = {
        content.period_number: idx for idx, content in enumerate(tkp.period_contents)
    }

    for period in tkp.teaching_plan.periods:
        if period.period_number not in claims_by_period:
            continue
        idx = period_to_index.get(period.period_number)
        if idx is None:
            continue

        reviewer_notes = "\n".join(f"- {item}" for item in claims_by_period[period.period_number])
        bundle = generate_period_bundle(
            period=period,
            knowledge=tkp.knowledge,
            metadata=tkp.metadata,
            doc=doc,
            review_notes=reviewer_notes,
        )
        tkp.period_contents[idx] = bundle.content
        if idx < len(tkp.activities):
            tkp.activities[idx] = bundle.activity
        if idx < len(tkp.assessments):
            tkp.assessments[idx] = bundle.assessment

    return tkp
