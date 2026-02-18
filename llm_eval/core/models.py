"""
Core data models for the evaluation framework.
"""

from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskType(str, Enum):
    """Supported task types."""
    QA = "qa"
    GENERATION = "generation"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    CODE = "code"
    SAFETY = "safety"
    DIALOGUE = "dialogue"


class JudgmentType(str, Enum):
    """Types of judgments."""
    PAIRWISE = "pairwise"
    ABSOLUTE = "absolute"
    RUBRIC = "rubric"


@dataclass
class Prompt:
    """Represents an evaluation prompt."""
    
    id: str
    text: str
    task_type: TaskType
    metadata: Dict[str, Any] = field(default_factory=dict)
    reference: Optional[str] = None  # Optional reference for certain tasks
    context: Optional[str] = None  # Additional context
    
    def __post_init__(self):
        """Validate prompt after initialization."""
        if not self.text.strip():
            raise ValueError("Prompt text cannot be empty")


@dataclass
class ModelResponse:
    """Represents a model's response to a prompt."""
    
    prompt_id: str
    model_id: str
    response_text: str
    response_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


@dataclass
class JudgeReasoning:
    """Structured reasoning from a judge."""
    
    criteria: str
    analysis: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    key_observations: List[str] = field(default_factory=list)


@dataclass
class Judgment:
    """A single judgment from a judge."""
    
    judge_id: str
    judgment_type: JudgmentType
    prompt_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # For pairwise comparison
    model_a_id: Optional[str] = None
    model_b_id: Optional[str] = None
    preference: Optional[Literal["A", "B", "tie"]] = None
    confidence: Optional[float] = None
    
    # For absolute scoring
    model_id: Optional[str] = None
    score: Optional[float] = None
    max_score: Optional[float] = None
    
    # For rubric-based
    rubric_scores: Optional[Dict[str, float]] = None
    
    # Common fields
    reasoning: Optional[JudgeReasoning] = None
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate judgment completeness."""
        if self.judgment_type == JudgmentType.PAIRWISE:
            return all([self.model_a_id, self.model_b_id, self.preference])
        elif self.judgment_type == JudgmentType.ABSOLUTE:
            return all([self.model_id, self.score is not None])
        elif self.judgment_type == JudgmentType.RUBRIC:
            return all([self.model_id, self.rubric_scores])
        return False


@dataclass
class EvaluationResult:
    """Final evaluation results."""
    
    prompt_id: str
    judgments: List[Judgment]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Aggregated scores
    model_scores: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, tuple] = field(default_factory=dict)
    
    # Statistical metrics
    inter_judge_agreement: Optional[float] = None
    agreement_metric: Optional[str] = None
    
    # Metadata
    evaluation_time: Optional[float] = None
    total_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_winner(self) -> Optional[str]:
        """Get the winning model ID."""
        if not self.model_scores:
            return None
        return max(self.model_scores.items(), key=lambda x: x[1])[0]
    
    def get_ranking(self) -> List[tuple]:
        """Get models ranked by score."""
        return sorted(self.model_scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class BatchEvaluationResult:
    """Results for a batch of evaluations."""
    
    results: List[EvaluationResult]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Aggregate statistics
    overall_rankings: Dict[str, float] = field(default_factory=dict)
    win_rates: Dict[str, float] = field(default_factory=dict)
    average_agreement: Optional[float] = None
    
    # Cost tracking
    total_cost: Optional[float] = None
    cost_per_evaluation: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_overall_rankings(self) -> Dict[str, float]:
        """Compute overall model rankings across all prompts."""
        model_scores: Dict[str, List[float]] = {}
        
        for result in self.results:
            for model_id, score in result.model_scores.items():
                if model_id not in model_scores:
                    model_scores[model_id] = []
                model_scores[model_id].append(score)
        
        # Average scores
        self.overall_rankings = {
            model_id: sum(scores) / len(scores)
            for model_id, scores in model_scores.items()
        }
        
        return self.overall_rankings
