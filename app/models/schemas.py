# 
# Data contract for the Teacher Knowledge Package (TKP) pipeline.
# 
# Every stage reads from and writes into this shared structure. Defining it
# up front, before any stage logic, is what makes the pipeline modular:
# each stage is a pure function of (previous TKP state) -> (updated TKP state),
# which is also what makes validation (Stage 9) and progress streaming
# straightforward.
# 

from __future__ import annotations
from enum import Enum
from typing import Optional, Union, get_origin, get_args
try:
    from types import UnionType
except ImportError:
    UnionType = None

from pydantic import BaseModel, Field, model_validator


class SafeBaseModel(BaseModel):
    """
    Base model that automatically coerces None list values to empty lists ([]).
    This prevents validation errors when the LLM returns null or omits list fields.
    """
    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, data):
        if not isinstance(data, dict):
            return data
        for name, field in cls.model_fields.items():
            annotation = field.annotation
            is_list = False
            if get_origin(annotation) is list or annotation is list:
                is_list = True
            elif get_origin(annotation) in (Union, UnionType):
                for arg in get_args(annotation):
                    if get_origin(arg) is list or arg is list:
                        is_list = True
                        break
            if is_list:
                val = data.get(name)
                if val is None:
                    data[name] = []
        return data


class DocumentTypeHint(str, Enum):
    """
    User-supplied hint (Stage 0, pre-pipeline) used to route parsing.
    MOSTLY_TEXT and TEXT_WITH_TABLES skip the OCR/vision path entirely;
    SCANNED_PDF forces it; NOT_SURE falls back to automatic heuristic
    detection in document_intelligence.py.
    """
    MOSTLY_TEXT = "mostly_text"
    TEXT_WITH_TABLES = "text_with_tables"
    TEXT_WITH_DIAGRAMS = "text_with_diagrams"
    TEXT_WITH_EQUATIONS = "text_with_equations"
    SCANNED_PDF = "scanned_pdf"
    NOT_SURE = "not_sure"


class UserContext(SafeBaseModel):
    """
    Answers to the small set of clarifying questions asked before the
    pipeline runs. Every field is optional -- a teacher who just wants to
    upload and go should be able to skip straight past this, with the
    pipeline falling back to inferring everything from the document alone,
    exactly as it did before this field existed.
    """
    grade_override: Optional[str] = None
    teaching_objectives: Optional[str] = None
    teaching_style: Optional[str] = None  # e.g. "lecture-heavy", "activity-based"
    time_constraint_minutes: Optional[int] = None  # total instructional time available, if fixed
    curriculum_board: Optional[str] = None  # e.g. CBSE, ICSE, State Board
    document_type_hint: DocumentTypeHint = DocumentTypeHint.NOT_SURE


class PipelineStage(str, Enum):
    UPLOAD = "upload"
    DOCUMENT_INTELLIGENCE = "document-intelligence"
    EDUCATIONAL_CLASSIFICATION = "educational-classification"
    KNOWLEDGE_EXTRACTION = "knowledge-extraction"
    TEACHING_PLANNER = "teaching-planner"
    PERIOD_GENERATION = "period-generation"
    GAP_ANALYSIS = "gap-analysis"
    VALIDATION = "validation"
    GROUNDING_VALIDATION = "grounding-validation"
    PUBLISHING = "publishing"
    DONE = "done"
    FAILED = "failed"


# ---------- Stage 1: Document Intelligence ----------

class DocumentBlock(SafeBaseModel):
    """One structural unit extracted from the raw document."""
    block_type: str  # heading | paragraph | table | figure | equation
    level: Optional[int] = None  # heading level, if applicable
    text: str
    page: Optional[int] = None


class ParsedDocument(SafeBaseModel):
    source_filename: str
    raw_text: str
    blocks: list[DocumentBlock] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)  # page count, format, etc.


# ---------- Stage 2: Educational Classification ----------

class EducationalMetadata(SafeBaseModel):
    subject: str
    grade: str
    difficulty: str  # Beginner | Intermediate | Advanced
    topic: str
    chapter: Optional[str] = None
    category: str  # STEM | Humanities | Language | Other
    curriculum_board: Optional[str] = None
    language: str = "English"


# ---------- Stage 3: Knowledge Extraction ----------

class Concept(SafeBaseModel):
    name: str
    definition: str
    related_formulae: list[str] = Field(default_factory=list)


class KnowledgeGraph(SafeBaseModel):
    learning_objectives: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    applications: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)


# ---------- Stage 4: Teaching Planner ----------

class Period(SafeBaseModel):
    period_number: int
    duration_minutes: int
    title: str
    learning_objectives: list[str]
    concepts_covered: list[str]


class TeachingPlan(SafeBaseModel):
    total_periods: int
    periods: list[Period]


# ---------- Stage 5: Classroom Content Generation ----------

class PeriodContent(SafeBaseModel):
    period_number: int
    entry_ticket: str
    teacher_script: str
    blackboard_notes: str
    classroom_activities: list[str]
    checkpoint_questions: list[str]
    exit_ticket: str
    homework: str
    mentor_moment: str
    source_citations: list[str] = Field(default_factory=list)


# ---------- Stage 6: Activity Generation ----------

class Activity(SafeBaseModel):
    period_number: int
    activity_type: str  # demonstration | role_play | experiment | discussion
    title: str
    duration_minutes: int
    materials_needed: list[str]
    teacher_instructions: str
    success_criteria: str
    source_citations: list[str] = Field(default_factory=list)


# ---------- Stage 7: Assessment Generation ----------

class AssessmentQuestion(SafeBaseModel):
    question_type: str  # mcq | short_answer | long_answer | numerical
    question_text: str
    options: Optional[list[str]] = None  # for MCQ
    correct_answer: str
    rubric: Optional[str] = None
    source_citations: list[str] = Field(default_factory=list)


class Assessment(SafeBaseModel):
    period_number: int
    questions: list[AssessmentQuestion]


# ---------- Stage 8: Learning Gap Analysis ----------

class LearningGap(SafeBaseModel):
    misconception: str
    diagnostic_question: str
    severity: str  # low | medium | high
    remedial_action: str


# ---------- Stage 9b: Grounding Validation ----------

class UngroundedClaim(SafeBaseModel):
    location: str  # e.g. "Period 2 teacher_script"
    claim: str
    reason: str  # why this doesn't trace back to the source document


class GroundingReport(SafeBaseModel):
    passed: bool
    ungrounded_claims: list[UngroundedClaim] = Field(default_factory=list)


# ---------- Stage 9: Validation ----------

class ValidationIssue(SafeBaseModel):
    stage: str
    severity: str  # warning | error
    message: str


class ValidationReport(SafeBaseModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


# ---------- Master Package ----------

class TeacherKnowledgePackage(SafeBaseModel):
    job_id: str
    source_filename: str
    user_context: Optional[UserContext] = None
    metadata: Optional[EducationalMetadata] = None
    knowledge: Optional[KnowledgeGraph] = None
    teaching_plan: Optional[TeachingPlan] = None
    period_contents: list[PeriodContent] = Field(default_factory=list)
    activities: list[Activity] = Field(default_factory=list)
    assessments: list[Assessment] = Field(default_factory=list)
    learning_gaps: list[LearningGap] = Field(default_factory=list)
    validation: Optional[ValidationReport] = None
    grounding: Optional[GroundingReport] = None
    stage: PipelineStage = PipelineStage.UPLOAD
    progress: int = 0


class PeriodGenerationBundle(SafeBaseModel):
    """Combined output for one period: content, activity, and assessment."""
    content: PeriodContent
    activity: Activity
    assessment: Assessment
