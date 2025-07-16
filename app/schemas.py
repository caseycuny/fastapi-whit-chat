from pydantic import BaseModel, Field, RootModel, ValidationError, conlist, confloat, conint
from typing import List, Dict, Optional, Union, Any
import re
from pydantic import validator

def normalize_key(key: str) -> str:
    """Normalize a key by removing special characters and converting to lowercase"""
    return re.sub(r'[^a-z0-9]', '', key.lower())

class RubricAverages(BaseModel):
    categories: List[str] = Field(..., description="List of rubric category names")
    values: List[Union[float, None]] = Field(..., description="List of average scores for each category")

class WritingPersonas(BaseModel):
    distribution: Dict[str, int] = Field(..., description="Distribution of writing personas")

class WeaknessItem(BaseModel):
    issue: Optional[str] = None
    weakness: Optional[str] = None  # Allow either field name
    frequency: Optional[int] = None
    count: Optional[int] = None  # Allow either field name
    examples: Optional[List[str]] = None  # Make examples optional

    def get_weakness_text(self) -> str:
        """Get the weakness text regardless of field name"""
        return self.issue or self.weakness or ""

    def get_frequency(self) -> int:
        """Get the frequency regardless of field name"""
        return self.frequency or self.count or 1

class GrammarIssues(RootModel):
    """Flexible model that can handle both list and nested dict formats"""
    root: Union[
        List[Dict[str, Any]],  # For list format
        Dict[str, Any]  # For nested dict format
    ]

class SentenceStructure(BaseModel):
    """Flexible model that accepts various field names"""
    short_sentences: Optional[str] = None
    short_sentence_ratio: Optional[str] = None
    long_sentences: Optional[str] = None
    long_sentence_ratio: Optional[str] = None
    variety: Optional[str] = None
    variety_score: Optional[str] = None
    patterns: Optional[List[str]] = None
    notable_examples: Optional[List[str]] = None

class VocabularyPatterns(BaseModel):
    advanced_word_choices: Optional[List[str]] = Field(None, alias="advanced_word_usage")
    advanced_word_usage: Optional[List[str]] = None
    repetitive_words: Optional[List[str]] = None
    colloquialisms_or_informal_language: Optional[List[str]] = None
    colloquialisms: Optional[List[str]] = None
    trends: Optional[List[str]] = None

class CognitiveSkillsAssessment(BaseModel):
    analysis: str
    synthesis: str
    evaluation: str

class ToneAnalysis(BaseModel):
    tone_shifts: Union[List[str], str]  # Accept either format
    emotional_appeals: List[str]
    formality_consistency: Optional[str] = None

class ClassTrendAnalysis(BaseModel):
    rubric_averages: RubricAverages
    writing_personas: WritingPersonas
    common_weaknesses_and_misconceptions: Dict[str, List[Dict[str, Union[str, int, List[str]]]]]
    instructional_blind_spots: List[str]
    common_strengths: List[str]
    sentence_structure_analysis: Dict[str, Union[bool, List[str]]]
    vocabulary_strength_patterns: Dict[str, Union[bool, List[str]]]
    notable_style_structure_patterns: List[str]
    cognitive_skills_assessment: Dict[str, str]
    grammar_and_syntax_issues_frequency: Dict[str, Union[int, List[str]]]
    tone_analysis_trends: Dict[str, Union[bool, List[str]]]

    class Config:
        strict = True
        extra = "forbid"
        populate_by_name = True
        allow_population_by_field_name = True

    @validator('rubric_averages')
    def validate_rubric_averages(cls, v):
        if len(v.categories) != len(v.values):
            raise ValueError("rubric_averages categories and values must have the same length")
        return v

    @validator('grammar_and_syntax_issues_frequency')
    def validate_grammar_issues(cls, v):
        for key, value in v.items():
            if key != "examples" and not isinstance(value, (int, float)):
                raise ValueError(f"grammar_and_syntax_issues_frequency.{key} must be a number")
        return v

    @validator('sentence_structure_analysis')
    def validate_sentence_structure(cls, v):
        for key, value in v.items():
            if key.endswith("_common") and not isinstance(value, bool):
                raise ValueError(f"sentence_structure_analysis.{key} must be a boolean")
        return v

    @validator('vocabulary_strength_patterns')
    def validate_vocabulary_patterns(cls, v):
        for key, value in v.items():
            if key.endswith("_common") and not isinstance(value, bool):
                raise ValueError(f"vocabulary_strength_patterns.{key} must be a boolean")
        return v

    @validator('tone_analysis_trends')
    def validate_tone_analysis(cls, v):
        for key, value in v.items():
            if key.endswith("_common") and not isinstance(value, bool):
                raise ValueError(f"tone_analysis_trends.{key} must be a boolean")
        return v

    def get_weaknesses(self) -> List[WeaknessItem]:
        """Get weaknesses regardless of field name"""
        return self.common_weaknesses_and_misconceptions.get(self.common_weaknesses_and_misconceptions.keys()[0], [])

class Topics(BaseModel):
    most_popular: List[str] = Field(default_factory=list, description="Topics that appeared most frequently in student responses")
    unique: List[str] = Field(default_factory=list, description="Topics that appeared only once")

class ElaborationTechniques(BaseModel):
    most_common: List[str] = Field(default_factory=list, description="Techniques most frequently observed")
    least_common_or_missing: List[str] = Field(default_factory=list, description="Techniques rarely or never observed")
    mixed_usage_observations: str = Field(
        default="", 
        description="Optional observations about technique usage patterns"
    )

class ClaimEvidenceReasoning(BaseModel):
    average_alignment_score: Optional[float] = Field(
        default=None, 
        description="Average alignment score if enough data is present"
    )
    reasoning_depth_summary: str = Field(
        default="", 
        description="Summary of reasoning depth observed"
    )
    evidence_language_reference_notes: str = Field(
        default="", 
        description="Notes about evidence language usage"
    )
    claim_elaboration_gaps: List[str] = Field(
        default_factory=list, 
        description="Common gaps in claim elaboration"
    )
    overgeneralizations: List[str] = Field(
        default_factory=list, 
        description="Common overgeneralizations observed"
    )

class LanguageUseAndStyle(BaseModel):
    rhetorical_verbs_common: List[str] = Field(
        default_factory=list, 
        description="Common rhetorical verbs observed"
    )
    causal_connectors_common: List[str] = Field(
        default_factory=list, 
        description="Common causal connectors observed"
    )
    metacognitive_phrases_common: List[str] = Field(
        default_factory=list, 
        description="Common metacognitive phrases observed"
    )

class ClasswideElaborationSummary(BaseModel):
    topics: Topics = Field(default_factory=Topics)
    elaboration_techniques: ElaborationTechniques = Field(default_factory=ElaborationTechniques)
    claim_evidence_reasoning: ClaimEvidenceReasoning = Field(default_factory=ClaimEvidenceReasoning)
    language_use_and_style: LanguageUseAndStyle = Field(default_factory=LanguageUseAndStyle)

    class Config:
        json_schema_extra = {
            "example": {
                "topics": {
                    "most_popular": ["Climate Change"],
                    "unique": ["Urban Planning"]
                },
                "elaboration_techniques": {
                    "most_common": ["Cause and Effect"],
                    "least_common_or_missing": ["Analogies", "Rhetorical Questions"],
                    "mixed_usage_observations": "Limited technique usage observed"
                },
                "claim_evidence_reasoning": {
                    "average_alignment_score": None,  # Not enough data for meaningful score
                    "reasoning_depth_summary": "Most responses showed basic cause-effect reasoning",
                    "evidence_language_reference_notes": "Limited evidence language observed",
                    "claim_elaboration_gaps": ["Brief responses"],
                    "overgeneralizations": []
                },
                "language_use_and_style": {
                    "rhetorical_verbs_common": ["shows"],
                    "causal_connectors_common": ["because"],
                    "metacognitive_phrases_common": []
                }
            }
        }

class GenerateArgumentParagraphResponse(BaseModel):
    step_1_title: str = Field(..., description="The label for the first step: 'Step 1: Let's Create a Claim'")
    claim: str = Field(..., description="The argumentative claim based on the provided topic")
    step_2_title: str = Field(..., description="The label for the second step, 'Step 2: Supporting Evidence'")
    evidence: str = Field(..., description="One sentence of supporting evidence in quotes including the proper citation")
    elaboration_prompt: str = Field(..., description="Instruction that leads into the final paragraph")
    full_paragraph: str = Field(..., description="The complete paragraph combining the claim and evidence with elaboration")

    class Config:
        extra = "forbid"
        strict = True

class ElaborationFeedbackResponse(BaseModel):
    strengths: list[str]
    areas_for_improvement: list[str]
    suggestions_for_elaboration: list[str]
    guiding_questions: list[str]
    praise_and_encouragement: list[str]
    full_paragraph: str

    class Config:
        extra = "forbid"
        strict = True

class ExemplarItem(BaseModel):
    category: str = Field(..., description="The rubric category this exemplar represents.")
    filename: str = Field(..., description="The filename of the student essay that is the exemplar for this category.")
    student_name: str = Field(..., description="The full name of the student whose essay is the exemplar for this category.")
    rationale: str = Field(..., description="A brief reason why this essay excels in this category.")
    excerpt: str = Field(..., description="A sentence or two from the student's writing (not a direct quote they used as evidence) that exemplifies excellence in this category.")

    class Config:
        extra = "forbid"
        strict = True

class ExemplarsResponse(BaseModel):
    exemplars: List[ExemplarItem]

    class Config:
        extra = "forbid"
        strict = True

def ensure_list(value) -> list:
    """Ensures a value is a list."""
    if value is None:
        return []
    if isinstance(value, str):
        # Handle simple comma-separated strings
        return [item.strip() for item in value.split(',') if item.strip()]
    return value if isinstance(value, list) else [value]

def ensure_float(value) -> Optional[float]:
    """Ensures a value is a float, returning None if conversion fails."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def clean_and_normalize_response(raw_data: dict) -> dict:
    """Cleans and normalizes raw dictionary data before Pydantic validation."""
    
    # 1. Normalize 'overall_score'
    if 'overall_score' in raw_data:
        raw_data['overall_score'] = ensure_float(raw_data['overall_score'])

    # 2. Normalize 'feedback' categories
    if 'feedback' in raw_data and 'categories' in raw_data['feedback']:
        for category in raw_data['feedback']['categories']:
            if 'score' in category:
                category['score'] = ensure_float(category['score'])

    # 3. Normalize 'sentence_structure_analysis'
    if 'sentence_structure_analysis' in raw_data:
        ssa = raw_data['sentence_structure_analysis']
        # Convert ratios and scores to strings as expected by the model
        for key in ['short_sentences_ratio', 'long_sentences_ratio', 'variety_score']:
            if key in ssa:
                ssa[key] = str(ssa[key])

    # 4. Normalize 'writing_persona'
    if 'writing_persona' in raw_data and not isinstance(raw_data['writing_persona'], dict):
        raw_data['writing_persona'] = {'type': 'Unknown', 'description': str(raw_data['writing_persona'])}
        
    return raw_data


class FeedbackCategory(BaseModel):
    name: str
    score: float
    strengths: List[str]
    areas_for_improvement: List[str]

class FeedbackObject(BaseModel):
    overall: str
    categories: List[FeedbackCategory]

class Excerpt(BaseModel):
    text: str
    comment: str

class GrammarAndSyntaxIssues(BaseModel):
    common_errors: List[str]
    examples: List[str]
    suggested_fixes: List[str]

class ToneAnalysis(BaseModel):
    tone_shifts: List[str]
    emotional_appeals: List[str]
    formality_consistency: str

class VocabularyStrength(BaseModel):
    advanced_word_choices: List[str]
    repetitive_words: List[str]
    colloquialisms_or_informal_language: List[str]

class SentenceStructureAnalysis(BaseModel):
    short_sentences_ratio: str
    long_sentences_ratio: str
    variety_score: str

class CognitiveSkills(BaseModel):
    analysis: str
    synthesis: str
    evaluation: str

class WritingPersona(BaseModel):
    type: str
    description: str

class ProcessEssayFeedbackResponse(BaseModel):
    overall_score: float
    feedback: FeedbackObject
    excerpts: List[Excerpt]
    grammar_and_syntax_issues: GrammarAndSyntaxIssues
    tone_analysis: ToneAnalysis
    vocabulary_strength: VocabularyStrength
    instructional_blind_spots: List[str]
    notable_patterns: List[str]
    sentence_structure_analysis: SentenceStructureAnalysis
    cognitive_skills: CognitiveSkills
    next_instructional_focus: List[str]
    writing_persona: WritingPersona

    class Config:
        extra = "forbid"
        strict = True

class SmartGroup(BaseModel):
    label: str = Field(..., description="Descriptive title for the group")
    students: List[str] = Field(..., description="List of student filenames in this group")

    class Config:
        extra = "forbid"
        strict = True

class SmartGroupingResponse(BaseModel):
    Smart_Strengths: List[SmartGroup] = Field(..., alias="Smart Strengths")
    Smart_Growth: List[SmartGroup] = Field(..., alias="Smart Growth")

    class Config:
        extra = "forbid"
        strict = True
        allow_population_by_field_name = True


# Lesson Plan Schemas
class KeyDesignPrinciples(BaseModel):
    growth_mindset: str
    udl: str
    brain_based_learning: str

    class Config:
        extra = "forbid"

class LessonPlan(BaseModel):
    title: str
    grade_level: str
    subject: str
    learning_objectives: List[str]
    warm_up: str
    mini_lesson: str
    guided_practice: str
    independent_practice: str
    formative_assessment: str
    closure_reflection: str
    materials: List[str]
    key_design_principles: KeyDesignPrinciples

    class Config:
        extra = "forbid"

class DictionImprovementSuggestion(BaseModel):
    word: str
    suggested_alternatives: list[str]

class ElaborationSummaryResponse(BaseModel):
    topic: str
    techniques_used: str
    ai_responsiveness: str
    strengths: list[str]
    areas_for_improvement: list[str]
    claim_evidence_reasoning: str
    language_use_and_style: str
    diction_improvement_suggestion: DictionImprovementSuggestion
    suggested_topics: list[str]
    
    class Config:
        extra = "forbid"
        strict = True

class ElaborationModelSentencesResponse(BaseModel):
    topic: str
    techniques: Dict[str, str]

    class Config:
        extra = "forbid"
        strict = True

class BloomPercentages(BaseModel):
    remember: confloat(ge=0, le=100)
    understand: confloat(ge=0, le=100)
    apply: confloat(ge=0, le=100)
    analyze: confloat(ge=0, le=100)
    evaluate: confloat(ge=0, le=100)
    create: confloat(ge=0, le=100)

class CategoryScores(BaseModel):
    argument_quality: confloat(ge=0, le=100)
    critical_thinking: confloat(ge=0, le=100)
    rhetorical_skill: confloat(ge=0, le=100)
    responsiveness: confloat(ge=0, le=100)
    structure_clarity: confloat(ge=0, le=100)
    style_delivery: confloat(ge=0, le=100)

class PersuasiveAppeal(BaseModel):
    appeal_type: str
    count: conint(ge=0)
    example_snippets: List[str]
    effectiveness_score: confloat(ge=0, le=100)

class RhetoricalDevice(BaseModel):
    device_type: str
    raw_label: str
    description: str
    count: conint(ge=0)
    example_snippets: List[str]
    effectiveness_score: confloat(ge=0, le=100)

class LogicalFallacy(BaseModel):
    fallacy_name: str
    raw_label: str
    description: str
    count: conint(ge=0)
    example_snippets: List[str]
    impact_score: confloat(ge=0, le=100)
    correction_suggestion: str

class ProcessDebateAnalysisResponse(BaseModel):
    overall_score: confloat(ge=0, le=100)
    ai_feedback_summary: str
    bloom_percentages: BloomPercentages
    category_scores: CategoryScores
    persuasive_appeals: List[PersuasiveAppeal]
    rhetorical_devices: List[RhetoricalDevice]

    class Config:
        extra = "forbid"
        strict = True

# Debate Insights Schemas
class GenerateDebateInsightsRequest(BaseModel):
    class_id: int
    assignment_id: int

class CachedDebateInsightsRequest(BaseModel):
    class_id: int
    assignment_id: int

class DebateInsightsResponse(BaseModel):
    student_notes_summary: str
    general_observations: str
    teaching_recommendations: str
    socratic_seminar_questions: List[str]

class GenerateDebateInsightsResponse(BaseModel):
    success: bool
    insights: Optional[DebateInsightsResponse] = None
    message: Optional[str] = None

class CachedDebateInsightsResponse(BaseModel):
    success: bool
    has_data: bool
    insights: Optional[DebateInsightsResponse] = None
    message: Optional[str] = None

# Exit Ticket Analysis Schemas
class ExitTicketAnalysisRequest(BaseModel):
    student_answer: str
    question: str
    correct_answer: str
    assignment_id: int
    student_id: int

class ExitTicketAnalysisResponse(BaseModel):
    success: bool
    correct: bool
    misconception_analysis: str
    misconception_type: str
    conceptual_vs_procedural: str
    intervention_suggestion: str
    error_severity: str
    message: Optional[str] = None


class ErrorSeverityDistribution(BaseModel):
    minor: int = Field(..., description="Number of students with minor errors")
    moderate: int = Field(..., description="Number of students with moderate errors")
    major: int = Field(..., description="Number of students with major errors")


class ConceptualProceduralBreakdown(BaseModel):
    conceptual: int = Field(..., description="Number of students with conceptual errors")
    procedural: int = Field(..., description="Number of students with procedural errors")
    both: int = Field(..., description="Number of students with both conceptual and procedural errors")


class InterventionNeed(BaseModel):
    intervention_type: str = Field(..., description="Type of intervention needed")
    student_count: int = Field(..., description="Number of students needing this intervention")


class StudentGroupingByMisconception(BaseModel):
    misconception_type: str = Field(..., description="The specific misconception type")
    student_count: int = Field(..., description="Number of students with this misconception")


class ExitTicketClasswideAnalysisData(BaseModel):
    """
    Pydantic model matching the OpenAI assistant schema for analyze_classwide_exit_tickets
    """
    mastery_percentage: float = Field(..., ge=0, le=100, description="Percentage of students who answered correctly (0-100)")
    class_readiness: str = Field(..., description="Overall class readiness based on mastery percentage")
    most_common_misconceptions: List[str] = Field(..., description="List of most common misconceptions ordered by frequency")
    error_severity_distribution: ErrorSeverityDistribution = Field(..., description="Distribution of error severity")
    conceptual_vs_procedural_breakdown: ConceptualProceduralBreakdown = Field(..., description="Breakdown of error types")
    similar_intervention_needs: List[InterventionNeed] = Field(..., description="Groups of students with similar intervention needs")
    priority_intervention_areas: List[str] = Field(..., description="Priority misconceptions to address first")
    reteaching_focus: str = Field(..., description="Specific concepts to reteach to whole class")
    clarity_issues: str = Field(..., description="Assessment of question clarity issues")
    student_groupings_by_misconception: List[StudentGroupingByMisconception] = Field(..., description="Students grouped by misconception types")
    success_indicators: str = Field(..., description="What successful students demonstrated")


class ExitTicketClasswideRequest(BaseModel):
    assignment_id: int = Field(..., description="Assignment ID for the exit ticket")


class ExitTicketClasswideResponse(BaseModel):
    success: bool
    analysis: ExitTicketClasswideAnalysisData
    total_responses: int = Field(..., description="Total number of student responses")
    completion_rate: Optional[float] = Field(None, description="Percentage of students who completed")
    thread_id: Optional[str] = Field(None, description="OpenAI thread ID for chat continuation")
    message: Optional[str] = None

# Mixed Groups Schemas
class MixedGroupsRequest(BaseModel):
    assignment_id: int = Field(..., description="Assignment ID to create mixed groups for")
    class_id: Optional[int] = Field(None, description="Optional class ID for additional context")

class MixedGroupStudent(BaseModel):
    name: str = Field(..., description="Student's full name")
    strength: str = Field(..., description="Key strength this student brings to the group")

class MixedGroup(BaseModel):
    group_number: int = Field(..., description="Group identifier (1, 2, 3, etc.)")
    focus_area: str = Field(..., description="Primary learning focus for this group")
    teaching_points: List[str] = Field(..., description="Specific teaching strategies for this group")
    students: List[MixedGroupStudent] = Field(..., description="Students assigned to this group")
    group_strategy: str = Field(..., description="Overall collaborative strategy for the group")

class MixedGroupsResponse(BaseModel):
    mixed_groups: List[MixedGroup] = Field(..., description="List of mixed ability groups")
    
    class Config:
        extra = "forbid"
        strict = True
        json_schema_extra = {
            "example": {
                "mixed_groups": [
                    {
                        "group_number": 1,
                        "focus_area": "Organization and Evidence Integration",
                        "teaching_points": [
                            "Strong organizers help those developing structure",
                            "Evidence experts guide citation techniques"
                        ],
                        "students": [
                            {"name": "Student A", "strength": "Excellent organization skills"},
                            {"name": "Student B", "strength": "Strong evidence usage"},
                            {"name": "Student C", "strength": "Developing writer eager to learn"}
                        ],
                        "group_strategy": "Peer mentoring with structured collaboration"
                    }
                ]
            }
        } 