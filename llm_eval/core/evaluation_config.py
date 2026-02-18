"""
Core evaluation configuration using Pydantic for type safety and validation.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class JudgeConfig(BaseModel):
    """Configuration for a single judge model."""
    
    model_name: str = Field(..., description="Name/identifier of the judge model")
    model_type: Literal["openai", "anthropic", "local", "custom"] = Field(
        ..., description="Type of model provider"
    )
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response")
    api_key: Optional[str] = Field(None, description="API key if required")
    endpoint: Optional[str] = Field(None, description="Custom endpoint if applicable")
    system_prompt: Optional[str] = Field(None, description="System prompt override")
    

class BiasConfig(BaseModel):
    """Configuration for bias mitigation strategies."""
    
    randomize_order: bool = Field(True, description="Randomize response order")
    use_blind_evaluation: bool = Field(True, description="Hide model identities")
    position_bias_correction: bool = Field(True, description="Apply position bias correction")
    length_normalization: bool = Field(True, description="Normalize for length bias")
    num_permutations: int = Field(2, ge=1, description="Number of order permutations")
    

class StatisticalConfig(BaseModel):
    """Configuration for statistical validation."""
    
    compute_confidence_intervals: bool = Field(True, description="Compute bootstrap CIs")
    confidence_level: float = Field(0.95, ge=0.0, le=1.0, description="CI level")
    num_bootstrap_samples: int = Field(1000, ge=100, description="Bootstrap iterations")
    compute_inter_judge_agreement: bool = Field(True, description="Compute agreement metrics")
    min_judge_agreement: float = Field(0.5, ge=0.0, le=1.0, description="Min agreement threshold")


class EvaluationConfig(BaseModel):
    """Master configuration for evaluation framework."""
    
    # Judge configuration
    judges: List[JudgeConfig] = Field(..., min_length=1, description="List of judge configurations")
    min_judges: int = Field(3, ge=1, description="Minimum number of judges required")
    
    # Evaluation strategy
    evaluation_method: Literal["pairwise", "absolute", "rubric", "hybrid"] = Field(
        "pairwise", description="Primary evaluation method"
    )
    
    # Task configuration
    task_type: Literal["qa", "generation", "summarization", "reasoning", "code", "safety", "dialogue"] = Field(
        ..., description="Type of task being evaluated"
    )
    
    # Bias mitigation
    bias_config: BiasConfig = Field(default_factory=BiasConfig)
    
    # Statistical validation
    statistical_config: StatisticalConfig = Field(default_factory=StatisticalConfig)
    
    # Scoring configuration
    scoring_method: Literal["bradley_terry", "elo", "simple_average", "weighted"] = Field(
        "bradley_terry", description="Score aggregation method"
    )
    
    # Safety and validation
    require_reasoning: bool = Field(True, description="Require judges to provide reasoning")
    validate_outputs: bool = Field(True, description="Validate judge outputs")
    timeout_seconds: int = Field(300, ge=1, description="Timeout for judge calls")
    
    # Cost optimization
    enable_caching: bool = Field(True, description="Enable response caching")
    max_cost_per_evaluation: Optional[float] = Field(None, description="Max cost budget")
    
    # Advanced options
    custom_rubric: Optional[Dict[str, Any]] = Field(None, description="Custom evaluation rubric")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator("judges")
    @classmethod
    def validate_judges(cls, v, info):
        """Ensure we have enough judges."""
        min_judges = info.data.get("min_judges", 3)
        if len(v) < min_judges:
            raise ValueError(f"At least {min_judges} judges required, got {len(v)}")
        return v
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"
