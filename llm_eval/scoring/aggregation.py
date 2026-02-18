"""
Score aggregation methods for combining judge verdicts.
"""

from typing import List, Dict
from collections import defaultdict

from llm_eval.core.models import Judgment, ModelResponse, JudgmentType


class ScoreAggregator:
    """
    Aggregates scores from multiple judges using various methods.
    
    Supports:
    - Simple averaging
    - Weighted averaging
    - Bradley-Terry model (via separate class)
    - Elo ratings (via separate class)
    """
    
    def __init__(self, method: str = "simple_average"):
        """
        Initialize aggregator.
        
        Args:
            method: Aggregation method (simple_average, weighted, bradley_terry, elo)
        """
        self.method = method
    
    def aggregate(
        self, judgments: List[Judgment], responses: List[ModelResponse]
    ) -> Dict[str, float]:
        """
        Aggregate judgments into final scores.
        
        Args:
            judgments: List of judgments from judges
            responses: List of model responses
            
        Returns:
            Dict mapping model_id to aggregated score
        """
        if not judgments:
            return {}
        
        # Determine judgment type
        judgment_type = judgments[0].judgment_type
        
        if judgment_type == JudgmentType.PAIRWISE:
            return self._aggregate_pairwise(judgments, responses)
        elif judgment_type == JudgmentType.ABSOLUTE:
            return self._aggregate_absolute(judgments)
        elif judgment_type == JudgmentType.RUBRIC:
            return self._aggregate_rubric(judgments)
        else:
            # Mixed types - combine methods
            return self._aggregate_mixed(judgments, responses)
    
    def _aggregate_pairwise(
        self, judgments: List[Judgment], responses: List[ModelResponse]
    ) -> Dict[str, float]:
        """Aggregate pairwise comparisons."""
        # Count wins, losses, and ties for each model
        model_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
        
        for judgment in judgments:
            if judgment.judgment_type != JudgmentType.PAIRWISE:
                continue
                
            model_a = judgment.model_a_id
            model_b = judgment.model_b_id
            preference = judgment.preference
            
            if preference == "A":
                model_stats[model_a]["wins"] += 1
                model_stats[model_b]["losses"] += 1
            elif preference == "B":
                model_stats[model_b]["wins"] += 1
                model_stats[model_a]["losses"] += 1
            elif preference == "tie":
                model_stats[model_a]["ties"] += 1
                model_stats[model_b]["ties"] += 1
        
        # Convert to scores (win rate with ties counting as 0.5)
        scores = {}
        for model_id, stats in model_stats.items():
            total = stats["wins"] + stats["losses"] + stats["ties"]
            if total > 0:
                scores[model_id] = (stats["wins"] + 0.5 * stats["ties"]) / total
            else:
                scores[model_id] = 0.5
        
        return scores
    
    def _aggregate_absolute(self, judgments: List[Judgment]) -> Dict[str, float]:
        """Aggregate absolute scores."""
        model_scores = defaultdict(list)
        
        for judgment in judgments:
            if judgment.judgment_type != JudgmentType.ABSOLUTE or judgment.score is None:
                continue
            
            # Normalize score to 0-1 range
            normalized_score = judgment.score / judgment.max_score if judgment.max_score else judgment.score
            model_scores[judgment.model_id].append(normalized_score)
        
        # Average scores for each model
        return {
            model_id: sum(scores) / len(scores)
            for model_id, scores in model_scores.items()
        }
    
    def _aggregate_rubric(self, judgments: List[Judgment]) -> Dict[str, float]:
        """Aggregate rubric-based scores."""
        model_scores = defaultdict(list)
        
        for judgment in judgments:
            if judgment.judgment_type != JudgmentType.RUBRIC or not judgment.rubric_scores:
                continue
            
            # Average across rubric criteria
            avg_score = sum(judgment.rubric_scores.values()) / len(judgment.rubric_scores)
            model_scores[judgment.model_id].append(avg_score)
        
        # Average scores for each model
        return {
            model_id: sum(scores) / len(scores)
            for model_id, scores in model_scores.items()
        }
    
    def _aggregate_mixed(
        self, judgments: List[Judgment], responses: List[ModelResponse]
    ) -> Dict[str, float]:
        """Aggregate mixed judgment types."""
        # Separate by type
        pairwise = [j for j in judgments if j.judgment_type == JudgmentType.PAIRWISE]
        absolute = [j for j in judgments if j.judgment_type == JudgmentType.ABSOLUTE]
        rubric = [j for j in judgments if j.judgment_type == JudgmentType.RUBRIC]
        
        # Aggregate each type
        scores = {}
        
        if pairwise:
            pairwise_scores = self._aggregate_pairwise(pairwise, responses)
            scores.update(pairwise_scores)
        
        if absolute:
            absolute_scores = self._aggregate_absolute(absolute)
            # Merge with existing scores
            for model_id, score in absolute_scores.items():
                if model_id in scores:
                    scores[model_id] = (scores[model_id] + score) / 2
                else:
                    scores[model_id] = score
        
        if rubric:
            rubric_scores = self._aggregate_rubric(rubric)
            # Merge with existing scores
            for model_id, score in rubric_scores.items():
                if model_id in scores:
                    scores[model_id] = (scores[model_id] + score) / 2
                else:
                    scores[model_id] = score
        
        return scores


class WeightedAggregator(ScoreAggregator):
    """
    Weighted score aggregation.
    
    Weights judges based on their historical accuracy or other metrics.
    """
    
    def __init__(self, judge_weights: Dict[str, float]):
        """
        Initialize weighted aggregator.
        
        Args:
            judge_weights: Dict mapping judge_id to weight
        """
        super().__init__(method="weighted")
        self.judge_weights = judge_weights
    
    def _aggregate_absolute(self, judgments: List[Judgment]) -> Dict[str, float]:
        """Aggregate absolute scores with weights."""
        model_scores = defaultdict(lambda: {"total": 0.0, "weight": 0.0})
        
        for judgment in judgments:
            if judgment.judgment_type != JudgmentType.ABSOLUTE or judgment.score is None:
                continue
            
            weight = self.judge_weights.get(judgment.judge_id, 1.0)
            normalized_score = judgment.score / judgment.max_score if judgment.max_score else judgment.score
            
            model_scores[judgment.model_id]["total"] += normalized_score * weight
            model_scores[judgment.model_id]["weight"] += weight
        
        # Compute weighted average
        return {
            model_id: data["total"] / data["weight"] if data["weight"] > 0 else 0.0
            for model_id, data in model_scores.items()
        }
