"""
Stage 10: Publishing.

Writes the final TeacherKnowledgePackage.json and renders consumable PDFs
(a Lesson Plan, a Teacher Guide, an Assessment Book) from the same
underlying data. Deliberately generates PDFs from the structured TKP data
rather than asking the LLM to write PDF-ready prose separately -- that
would risk the PDF drifting from the JSON, and it's strictly more work for
no benefit since every field needed for the PDFs already exists in the TKP.
"""

import json
import unicodedata
from pathlib import Path

from fpdf import FPDF

from app.models.schemas import TeacherKnowledgePackage


def publish_package(tkp: TeacherKnowledgePackage, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "TeacherKnowledgePackage.json"
    json_path.write_text(tkp.model_dump_json(indent=2), encoding="utf-8")

    lesson_plan_path = output_dir / "LessonPlan.pdf"
    _render_lesson_plan_pdf(tkp, lesson_plan_path)

    teacher_guide_path = output_dir / "TeacherGuide.pdf"
    _render_teacher_guide_pdf(tkp, teacher_guide_path)

    assessment_book_path = output_dir / "AssessmentBook.pdf"
    _render_assessment_book_pdf(tkp, assessment_book_path)

    return {
        "json": str(json_path),
        "lesson_plan_pdf": str(lesson_plan_path),
        "teacher_guide_pdf": str(teacher_guide_path),
        "assessment_book_pdf": str(assessment_book_path),
    }


def _cell(pdf: FPDF, h: int, text: str) -> None:
    """
    multi_cell in fpdf2 leaves the cursor at the right edge of the last
    line by default (new_x=RIGHT), not back at the left margin, so a
    width=0 call immediately after has almost no horizontal space left
    and raises FPDFException. Every multi_cell call in this module goes
    through here so that reset happens exactly once, in one place.
    """
    pdf.multi_cell(0, h, _pdf_safe(text))
    pdf.ln()


def _pdf_safe(text: str) -> str:
    """
    fpdf2 core fonts are Latin-1 only, so normalize user/model text into a
    PDF-safe form instead of letting exotic Unicode characters explode the
    export step on Windows or other locale-sensitive environments.
    """
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("→", "->")
        .replace("—", "-")
        .replace("–", "-")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("•", "-")
    )
    return text.encode("latin-1", "replace").decode("latin-1")


def _new_pdf(title: str) -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    _cell(pdf, 10, title)
    pdf.ln(2)
    return pdf


def _render_lesson_plan_pdf(tkp: TeacherKnowledgePackage, path: Path) -> None:
    subject = tkp.metadata.subject if tkp.metadata else "Unknown"
    pdf = _new_pdf(f"Lesson Plan: {subject}")
    if not tkp.teaching_plan:
        pdf.output(str(path))
        return

    for period in tkp.teaching_plan.periods:
        pdf.set_font("Helvetica", "B", 13)
        _cell(pdf, 8, f"Period {period.period_number}: {period.title} ({period.duration_minutes} min)")
        pdf.set_font("Helvetica", "", 11)
        _cell(pdf, 6, "Objectives: " + "; ".join(period.learning_objectives))
        _cell(pdf, 6, "Concepts: " + ", ".join(period.concepts_covered))
        pdf.ln(4)
    pdf.output(str(path))


def _render_teacher_guide_pdf(tkp: TeacherKnowledgePackage, path: Path) -> None:
    subject = tkp.metadata.subject if tkp.metadata else "Unknown"
    pdf = _new_pdf(f"Teacher Guide: {subject}")

    for pc in tkp.period_contents:
        pdf.set_font("Helvetica", "B", 13)
        _cell(pdf, 8, f"Period {pc.period_number}")
        pdf.set_font("Helvetica", "B", 11)
        _cell(pdf, 6, "Entry Ticket")
        pdf.set_font("Helvetica", "", 11)
        _cell(pdf, 6, pc.entry_ticket)
        pdf.set_font("Helvetica", "B", 11)
        _cell(pdf, 6, "Teacher Script")
        pdf.set_font("Helvetica", "", 11)
        _cell(pdf, 6, pc.teacher_script)
        pdf.set_font("Helvetica", "B", 11)
        _cell(pdf, 6, "Blackboard Notes")
        pdf.set_font("Helvetica", "", 11)
        _cell(pdf, 6, pc.blackboard_notes)
        pdf.set_font("Helvetica", "B", 11)
        _cell(pdf, 6, "Mentor Moment")
        pdf.set_font("Helvetica", "", 11)
        _cell(pdf, 6, pc.mentor_moment)
        pdf.set_font("Helvetica", "B", 11)
        _cell(pdf, 6, "Homework")
        pdf.set_font("Helvetica", "", 11)
        _cell(pdf, 6, pc.homework)
        pdf.ln(6)

    pdf.output(str(path))


def _render_assessment_book_pdf(tkp: TeacherKnowledgePackage, path: Path) -> None:
    subject = tkp.metadata.subject if tkp.metadata else "Unknown"
    pdf = _new_pdf(f"Assessment Book: {subject}")

    for assessment in tkp.assessments:
        pdf.set_font("Helvetica", "B", 13)
        _cell(pdf, 8, f"Period {assessment.period_number} Assessment")
        for i, q in enumerate(assessment.questions, start=1):
            pdf.set_font("Helvetica", "", 11)
            _cell(pdf, 6, f"{i}. [{q.question_type.upper()}] {q.question_text}")
            if q.options:
                for opt_i, opt in enumerate(q.options):
                    letter = chr(ord("A") + opt_i)
                    _cell(pdf, 6, f"    {letter}. {opt}")
            pdf.set_font("Helvetica", "I", 10)
            _cell(pdf, 6, f"    Answer: {q.correct_answer}")
            pdf.ln(2)
        pdf.ln(4)

    pdf.output(str(path))
