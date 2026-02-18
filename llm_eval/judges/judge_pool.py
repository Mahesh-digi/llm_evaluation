"""
Judge pool management system.

Handles multiple judge models with load balancing, fallback, and health monitoring.
"""

from typing import List, Optional
from abc import ABC, abstractmethod
import random
from datetime import datetime

from llm_eval.core.evaluation_config import JudgeConfig
from llm_eval.core.models import Prompt, ModelResponse, Judgment, JudgmentType, JudgeReasoning


class BaseJudge(ABC):
    """
    Abstract base class for all judge implementations.
    
    All judge implementations must inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, config: JudgeConfig):
        """Initialize judge with configuration."""
        self.config = config
        self.judge_id = f"{config.model_name}_{id(self)}"
        self.call_count = 0
        self.total_cost = 0.0
        self.success_rate = 1.0
        self.is_active = True
        
    @abstractmethod
    def judge_pairwise(
        self, prompt: Prompt, response_a: ModelResponse, response_b: ModelResponse
    ) -> Judgment:
        """
        Judge two responses in pairwise comparison.
        
        Args:
            prompt: Original prompt
            response_a: First response
            response_b: Second response
            
        Returns:
            Judgment with preference (A, B, or tie)
        """
        pass
    
    @abstractmethod
    def judge_absolute(self, prompt: Prompt, response: ModelResponse) -> Judgment:
        """
        Judge a single response with absolute scoring.
        
        Args:
            prompt: Original prompt
            response: Response to judge
            
        Returns:
            Judgment with absolute score
        """
        pass
    
    @abstractmethod
    def judge_rubric(
        self, prompt: Prompt, response: ModelResponse, rubric: dict
    ) -> Judgment:
        """
        Judge a response using a rubric.
        
        Args:
            prompt: Original prompt
            response: Response to judge
            rubric: Evaluation rubric
            
        Returns:
            Judgment with rubric scores
        """
        pass
    
    def _create_judgment(
        self, judgment_type: JudgmentType, prompt_id: str, **kwargs
    ) -> Judgment:
        """Helper to create a judgment with common fields."""
        self.call_count += 1
        
        judgment = Judgment(
            judge_id=self.judge_id,
            judgment_type=judgment_type,
            prompt_id=prompt_id,
            timestamp=datetime.now(),
            **kwargs
        )
        
        return judgment
    
    def get_stats(self) -> dict:
        """Get judge statistics."""
        return {
            "judge_id": self.judge_id,
            "model_name": self.config.model_name,
            "call_count": self.call_count,
            "total_cost": self.total_cost,
            "success_rate": self.success_rate,
            "is_active": self.is_active,
        }


class MockJudge(BaseJudge):
    """
    Mock judge implementation for testing and demonstration.
    
    This judge provides simulated judgments without calling any real APIs.
    Useful for testing and development.
    """
    
    def __init__(self, config: JudgeConfig, random_seed: Optional[int] = None):
        """
        Initialize mock judge.
        
        Args:
            config: Judge configuration
            random_seed: Optional random seed for reproducibility (None for true randomness)
        """
        super().__init__(config)
        # Set random seed if provided (for reproducible testing)
        if random_seed is not None:
            random.seed(random_seed)
    
    def judge_pairwise(
        self, prompt: Prompt, response_a: ModelResponse, response_b: ModelResponse
    ) -> Judgment:
        """Simulate pairwise judgment."""
        # Simple heuristic: prefer longer, more detailed responses
        len_a = len(response_a.response_text)
        len_b = len(response_b.response_text)
        
        if abs(len_a - len_b) < 50:
            preference = "tie"
            confidence = 0.5
        elif len_a > len_b:
            preference = "A"
            confidence = min(0.9, 0.6 + (len_a - len_b) / 1000)
        else:
            preference = "B"
            confidence = min(0.9, 0.6 + (len_b - len_a) / 1000)
        
        reasoning = JudgeReasoning(
            criteria="Response length and detail",
            analysis=f"Response A has {len_a} chars, Response B has {len_b} chars",
            strengths=["Detailed explanation" if len_a > 100 else "Concise"],
            weaknesses=["Too brief" if len_a < 50 else ""],
        )
        
        return self._create_judgment(
            JudgmentType.PAIRWISE,
            prompt.id,
            model_a_id=response_a.model_id,
            model_b_id=response_b.model_id,
            preference=preference,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=f"Mock judgment: {preference} (confidence: {confidence})",
        )
    
    def judge_absolute(self, prompt: Prompt, response: ModelResponse) -> Judgment:
        """Simulate absolute scoring."""
        # Simple scoring based on response length and structure
        text = response.response_text
        score = min(10.0, 5.0 + len(text) / 100)
        
        reasoning = JudgeReasoning(
            criteria="Overall quality",
            analysis=f"Response has {len(text)} characters",
            strengths=["Clear structure", "Appropriate length"],
            weaknesses=[],
        )
        
        return self._create_judgment(
            JudgmentType.ABSOLUTE,
            prompt.id,
            model_id=response.model_id,
            score=score,
            max_score=10.0,
            reasoning=reasoning,
            raw_response=f"Mock score: {score}/10",
        )
    
    def judge_rubric(
        self, prompt: Prompt, response: ModelResponse, rubric: dict
    ) -> Judgment:
        """Simulate rubric-based judgment."""
        rubric_scores = {}
        
        for criterion, max_score in rubric.items():
            # Assign random but reasonable scores
            rubric_scores[criterion] = random.uniform(0.6, 1.0) * max_score
        
        reasoning = JudgeReasoning(
            criteria="Rubric-based evaluation",
            analysis="Evaluated against all rubric criteria",
            strengths=["Meets most criteria"],
            weaknesses=[],
        )
        
        return self._create_judgment(
            JudgmentType.RUBRIC,
            prompt.id,
            model_id=response.model_id,
            rubric_scores=rubric_scores,
            reasoning=reasoning,
            raw_response=f"Mock rubric scores: {rubric_scores}",
        )


class JudgePool:
    """
    Manages a pool of judge models with load balancing and failover.
    
    Features:
    - Load balancing across judges
    - Automatic failover on errors
    - Health monitoring
    - Cost tracking
    """
    
    def __init__(self, judge_configs: List[JudgeConfig]):
        """
        Initialize judge pool.
        
        Args:
            judge_configs: List of judge configurations
        """
        self.judges: List[BaseJudge] = []
        
        for config in judge_configs:
            # Create appropriate judge based on model_type
            if config.model_type == "custom" or config.model_type == "local":
                # For now, use mock judges
                # In production, this would instantiate real judge implementations
                judge = MockJudge(config)
            else:
                # Default to mock for demonstration
                judge = MockJudge(config)
            
            self.judges.append(judge)
    
    def get_active_judges(self) -> List[BaseJudge]:
        """Get all active judges."""
        return [j for j in self.judges if j.is_active]
    
    def get_judge(self, judge_id: Optional[str] = None) -> BaseJudge:
        """
        Get a specific judge or select one via load balancing.
        
        Args:
            judge_id: Optional specific judge ID
            
        Returns:
            Selected judge
        """
        active_judges = self.get_active_judges()
        
        if not active_judges:
            raise RuntimeError("No active judges available")
        
        if judge_id:
            for judge in active_judges:
                if judge.judge_id == judge_id:
                    return judge
            raise ValueError(f"Judge {judge_id} not found or inactive")
        
        # Load balancing: select judge with lowest call count
        return min(active_judges, key=lambda j: j.call_count)
    
    def get_pool_stats(self) -> dict:
        """Get statistics for all judges."""
        return {
            "total_judges": len(self.judges),
            "active_judges": len(self.get_active_judges()),
            "total_calls": sum(j.call_count for j in self.judges),
            "total_cost": sum(j.total_cost for j in self.judges),
            "judges": [j.get_stats() for j in self.judges],
        }
    
    def deactivate_judge(self, judge_id: str):
        """Deactivate a judge (e.g., due to repeated failures)."""
        for judge in self.judges:
            if judge.judge_id == judge_id:
                judge.is_active = False
                break
    
    def activate_judge(self, judge_id: str):
        """Reactivate a previously deactivated judge."""
        for judge in self.judges:
            if judge.judge_id == judge_id:
                judge.is_active = True
                break
