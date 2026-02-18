"""
Bradley-Terry model for pairwise comparison aggregation.

The Bradley-Terry model estimates latent "skill" scores from pairwise
comparison data, providing a principled statistical approach to ranking.
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

from llm_eval.core.models import Judgment, JudgmentType


class BradleyTerryModel:
    """
    Bradley-Terry model for pairwise comparison analysis.
    
    The model estimates skill parameters θ_i for each model such that
    the probability of model i beating model j is:
        P(i > j) = θ_i / (θ_i + θ_j)
    
    We use iterative algorithms (MM or Newton) to estimate parameters.
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize Bradley-Terry model.
        
        Args:
            max_iterations: Maximum iterations for fitting
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.model_scores: Dict[str, float] = {}
    
    def fit(self, judgments: List[Judgment]) -> Dict[str, float]:
        """
        Fit Bradley-Terry model to pairwise judgments.
        
        Args:
            judgments: List of pairwise judgments
            
        Returns:
            Dict mapping model_id to estimated skill score
        """
        # Extract pairwise comparison data
        comparisons = self._extract_comparisons(judgments)
        
        if not comparisons:
            return {}
        
        # Get unique models
        models = sorted(set(m for pair in comparisons for m in pair[:2]))
        model_to_idx = {m: i for i, m in enumerate(models)}
        n_models = len(models)
        
        # Count wins for each pair
        win_matrix = np.zeros((n_models, n_models))
        
        for model_a, model_b, winner in comparisons:
            i, j = model_to_idx[model_a], model_to_idx[model_b]
            if winner == model_a:
                win_matrix[i, j] += 1
            elif winner == model_b:
                win_matrix[j, i] += 1
            else:  # tie
                win_matrix[i, j] += 0.5
                win_matrix[j, i] += 0.5
        
        # Initialize parameters (log scale)
        log_theta = np.zeros(n_models)
        
        # MM algorithm for fitting
        for iteration in range(self.max_iterations):
            theta = np.exp(log_theta)
            
            # Compute updates
            wins = win_matrix.sum(axis=1)  # Total wins for each model
            
            # Compute denominators
            denominators = np.zeros(n_models)
            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        n_comparisons = win_matrix[i, j] + win_matrix[j, i]
                        if n_comparisons > 0:
                            denominators[i] += n_comparisons / (theta[i] + theta[j])
            
            # Update
            new_theta = wins / np.maximum(denominators, 1e-10)
            
            # Normalize to prevent overflow
            new_theta = new_theta / np.mean(new_theta)
            
            # Check convergence
            if np.max(np.abs(new_theta - theta)) < self.tolerance:
                break
            
            log_theta = np.log(new_theta)
        
        # Convert to dict
        theta = np.exp(log_theta)
        self.model_scores = {models[i]: float(theta[i]) for i in range(n_models)}
        
        return self.model_scores
    
    def _extract_comparisons(self, judgments: List[Judgment]) -> List[Tuple[str, str, str]]:
        """
        Extract comparison tuples from judgments.
        
        Returns:
            List of (model_a, model_b, winner) tuples
        """
        comparisons = []
        
        for judgment in judgments:
            if judgment.judgment_type != JudgmentType.PAIRWISE:
                continue
            
            model_a = judgment.model_a_id
            model_b = judgment.model_b_id
            
            if judgment.preference == "A":
                winner = model_a
            elif judgment.preference == "B":
                winner = model_b
            else:  # tie
                winner = "tie"
            
            comparisons.append((model_a, model_b, winner))
        
        return comparisons
    
    def predict_win_probability(self, model_a: str, model_b: str) -> float:
        """
        Predict probability of model_a beating model_b.
        
        Args:
            model_a: First model ID
            model_b: Second model ID
            
        Returns:
            Probability of model_a winning
        """
        if model_a not in self.model_scores or model_b not in self.model_scores:
            return 0.5
        
        theta_a = self.model_scores[model_a]
        theta_b = self.model_scores[model_b]
        
        return theta_a / (theta_a + theta_b)
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """
        Get models ranked by estimated skill.
        
        Returns:
            List of (model_id, score) tuples, sorted by score descending
        """
        return sorted(self.model_scores.items(), key=lambda x: x[1], reverse=True)
