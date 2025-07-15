# fastapi-whit-chat/app/models.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .db import Base
from datetime import datetime

class CustomUser(Base):
    __tablename__ = "jarvis_app_customuser"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    role = Column(String)
    school = Column(String)
    grade_level = Column(String)
    subject_level = Column(String)
    district = Column(String)
    city = Column(String)
    state = Column(String)
    # Relationships omitted for brevity

class Class(Base):
    __tablename__ = "jarvis_app_class"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    teacher_id = Column(Integer, ForeignKey("jarvis_app_customuser.id"))
    period = Column(String)
    year = Column(Integer)
    grade = Column(String)
    class_code = Column(String, unique=True)
    created_at = Column(DateTime)
    # Relationships omitted for brevity

class Assignment(Base):
    __tablename__ = "jarvis_app_assignment"
    id = Column(Integer, primary_key=True, index=True)
    class_instance_id = Column(Integer, ForeignKey("jarvis_app_class.id"))
    title = Column(String)
    description = Column(Text)
    due_date = Column(DateTime)
    rubric_type = Column(String)
    type = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    last_analysis_run = Column(DateTime)
    is_active = Column(Boolean)
    is_graded = Column(Boolean)
    average_score = Column(Float)
    ai_feedback_strengths = Column(Boolean)
    ai_feedback_improvements = Column(Boolean)
    ai_feedback_grammar = Column(Boolean)
    ai_feedback_tone = Column(Boolean)
    ai_feedback_vocabulary = Column(Boolean)
    ai_feedback_sentence_structure = Column(Boolean)
    ai_feedback_overall_score = Column(Boolean)
    ai_feedback_category_scores = Column(Boolean)

class Submission(Base):
    __tablename__ = "jarvis_app_submission"
    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer, ForeignKey("jarvis_app_assignment.id"))
    student_id = Column(Integer, ForeignKey("jarvis_app_customuser.id"))
    file = Column(String)
    status = Column(String)
    feedback = Column(JSON)
    score = Column(Float)
    submitted_at = Column(DateTime)
    teacher_score = Column(Float)
    teacher_feedback = Column(Text)
    teacher_text_comment = Column(Text)
    teacher_graded_at = Column(DateTime)
    teacher_grading_status = Column(String)
    audio_feedback = Column(String)
    teacher_category_scores = Column(JSON)
    student = relationship("CustomUser", backref="submissions")

class ChatThread(Base):
    __tablename__ = "jarvis_app_chatthread"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("jarvis_app_customuser.id"))
    assignment_id = Column(Integer, ForeignKey("jarvis_app_assignment.id"), nullable=True)
    thread_id = Column(String)
    title = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    is_active = Column(Boolean)

class ChatMessage(Base):
    __tablename__ = "jarvis_app_chatmessage"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey("jarvis_app_chatthread.id"))
    sender = Column(String)
    text = Column(Text)
    created_at = Column(DateTime)

class DebateTopic(Base):
    __tablename__ = "jarvis_app_debatetopic"
    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("jarvis_app_customuser.id"))
    topic = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    assignment_id = Column(Integer, ForeignKey("jarvis_app_assignment.id"))

class DebatePrompt(Base):
    __tablename__ = "jarvis_app_debateprompt"
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("jarvis_app_debatetopic.id"))
    assignment_id = Column(Integer, ForeignKey("jarvis_app_assignment.id"))
    text = Column(Text)
    is_ai_generated = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class StudentDebate(Base):
    __tablename__ = "jarvis_app_studentdebate"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("jarvis_app_customuser.id"))
    prompt_id = Column(Integer, ForeignKey("jarvis_app_debateprompt.id"), nullable=True)
    topic_id = Column(Integer, ForeignKey("jarvis_app_debatetopic.id"), nullable=True)
    assignment_id = Column(Integer, ForeignKey("jarvis_app_assignment.id"), nullable=True)
    chosen_side = Column(String(10))  # 'pro' or 'con'
    transcript = Column(Text)
    submitted_at = Column(DateTime, default=datetime.utcnow)

class DebateAnalysis(Base):
    __tablename__ = "jarvis_app_debateanalysis"
    id = Column(Integer, primary_key=True, index=True)
    student_debate_id = Column(Integer, ForeignKey("jarvis_app_studentdebate.id"))
    transcript_text = Column(Text)
    overall_score = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Bloom's level percentages
    remember_pct = Column(Float)
    understand_pct = Column(Float)
    apply_pct = Column(Float)
    analyze_pct = Column(Float)
    evaluate_pct = Column(Float)
    create_pct = Column(Float)
    
    # Debate score by category
    argument_quality = Column(Integer)
    critical_thinking = Column(Integer)
    rhetorical_skill = Column(Integer)
    responsiveness = Column(Integer)
    structure_clarity = Column(Integer)
    style_delivery = Column(Integer, nullable=True)
    
    # Additional metadata
    word_count = Column(Integer, default=0)
    speaking_time_seconds = Column(Integer, default=0)
    average_response_time = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    ai_feedback_summary = Column(Text, nullable=True)

class PersuasiveAppeal(Base):
    __tablename__ = "jarvis_app_persuasiveappeal"
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("jarvis_app_debateanalysis.id"))
    appeal_type = Column(String(20))  # 'ethos', 'pathos', 'logos'
    count = Column(Integer)
    example_snippets = Column(JSON, default=list)
    effectiveness_score = Column(Integer, nullable=True)

class RhetoricalDevice(Base):
    __tablename__ = "jarvis_app_rhetoricaldevice"
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("jarvis_app_debateanalysis.id"))
    device_type = Column(String(50))
    raw_label = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    count = Column(Integer)
    example_snippets = Column(JSON, default=list)
    effectiveness_score = Column(Integer, nullable=True)

class ClasswideDebateData(Base):
    __tablename__ = "jarvis_app_classwidedebatedata"
    id = Column(Integer, primary_key=True, index=True)
    class_instance_id = Column(Integer, ForeignKey("jarvis_app_class.id"))
    assignment_id = Column(Integer, ForeignKey("jarvis_app_assignment.id"))
    
    # Performance metrics
    total_debates_analyzed = Column(Integer, default=0)
    overall_score_avg = Column(Float, nullable=True)
    argument_quality_avg = Column(Float, nullable=True)
    critical_thinking_avg = Column(Float, nullable=True)
    rhetorical_skill_avg = Column(Float, nullable=True)
    responsiveness_avg = Column(Float, nullable=True)
    structure_clarity_avg = Column(Float, nullable=True)
    
    # Bloom's taxonomy averages
    remember_pct_avg = Column(Float, nullable=True)
    understand_pct_avg = Column(Float, nullable=True)
    apply_pct_avg = Column(Float, nullable=True)
    analyze_pct_avg = Column(Float, nullable=True)
    evaluate_pct_avg = Column(Float, nullable=True)
    create_pct_avg = Column(Float, nullable=True)
    
    # Debate metrics
    word_count_avg = Column(Float, nullable=True)
    speaking_time_avg = Column(Float, nullable=True)
    response_time_avg = Column(Float, nullable=True)
    confidence_score_avg = Column(Float, nullable=True)
    
    # Persuasive appeals and rhetorical devices (JSON)
    persuasive_appeals_data = Column(JSON, default=dict)
    rhetorical_devices_data = Column(JSON, default=list)
    prompt_analytics = Column(JSON, default=dict)
    
    # AI-generated insights
    student_notes_summary = Column(Text, nullable=True)
    general_observations = Column(Text, nullable=True)
    teaching_recommendations = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DebateNoteToTeacher(Base):
    __tablename__ = "jarvis_app_debatenotetoteacher"
    id = Column(Integer, primary_key=True, index=True)
    student_debate_id = Column(Integer, ForeignKey("jarvis_app_studentdebate.id"))
    note = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

class FeedbackCategory(Base):
    __tablename__ = "jarvis_app_feedbackcategory"
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("jarvis_app_submission.id"))
    name = Column(String)
    score = Column(Integer)
    strengths = Column(JSON)
    areas_for_improvement = Column(JSON)

class NextInstructionalFocus(Base):
    __tablename__ = "jarvis_app_nextinstructionalfocus"
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("jarvis_app_submission.id"))
    focus = Column(String)

class InstructionalBlindSpot(Base):
    __tablename__ = "jarvis_app_instructionalblindspot"
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("jarvis_app_submission.id"))
    blind_spot = Column(String)

class WritingPersona(Base):
    __tablename__ = "jarvis_app_writingpersona"
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("jarvis_app_submission.id"))
    type = Column(String)
    description = Column(String)

class CognitiveSkill(Base):
    __tablename__ = "jarvis_app_cognitiveskill"
    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("jarvis_app_submission.id"))
    analysis = Column(String)
    synthesis = Column(String)
    evaluation = Column(String)

# Add more models as needed for your FastAPI use cases, following the same pattern.