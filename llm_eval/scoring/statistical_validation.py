"""
Statistical validation and reliability metrics.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

from llm_eval.core.models import Judgment, JudgmentType
from llm_eval.core.evaluation_config import StatisticalConfig


class StatisticalValidator:
    """
    Provides statistical validation and reliability metrics.
    
    Features:
    - Bootstrap confidence intervals
    - Inter-judge agreement (Cohen's kappa, Fleiss' kappa)
    - Krippendorff's alpha
    - Stability testing
    """
    
    def __init__(self, config: StatisticalConfig):
        """
        Initialize validator.
        
        Args:
            config: Statistical configuration
        """
        self.config = config
    
    def compute_confidence_intervals(
        self, judgments: List[Judgment], point_estimates: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for model scores.
        
        Args:
            judgments: List of judgments
            point_estimates: Point estimates for each model
            
        Returns:
            Dict mapping model_id to (lower_bound, upper_bound) tuple
        """
        if not self.config.compute_confidence_intervals:
            return {}
        
        confidence_intervals = {}
        
        # Bootstrap resampling
        n_judgments = len(judgments)
        bootstrap_scores = defaultdict(list)
        
        for _ in range(self.config.num_bootstrap_samples):
            # Resample judgments with replacement
            resampled_indices = np.random.choice(
                n_judgments, size=n_judgments, replace=True
            )
            resampled_judgments = [judgments[i] for i in resampled_indices]
            
            # Recompute scores
            scores = self._compute_scores_from_judgments(resampled_judgments)
            
            for model_id, score in scores.items():
                bootstrap_scores[model_id].append(score)
        
        # Compute confidence intervals
        alpha = 1 - self.config.confidence_level
        
        for model_id in point_estimates.keys():
            if model_id in bootstrap_scores:
                scores = np.array(bootstrap_scores[model_id])
                lower = np.percentile(scores, 100 * alpha / 2)
                upper = np.percentile(scores, 100 * (1 - alpha / 2))
                confidence_intervals[model_id] = (float(lower), float(upper))
            else:
                # No bootstrap samples - use point estimate
                confidence_intervals[model_id] = (
                    point_estimates[model_id],
                    point_estimates[model_id],
                )
        
        return confidence_intervals
    
    def compute_inter_judge_agreement(
        self, judgments: List[Judgment]
    ) -> Tuple[float, str]:
        """
        Compute inter-judge agreement metrics.
        
        Args:
            judgments: List of judgments
            
        Returns:
            Tuple of (agreement_score, metric_name)
        """
        if not self.config.compute_inter_judge_agreement:
            return 0.0, "none"
        
        # Group judgments by prompt and comparison
        grouped = self._group_judgments(judgments)
        
        if len(grouped) == 0:
            return 0.0, "none"
        
        # Determine which metric to use based on judgment type
        judgment_type = judgments[0].judgment_type
        
        if judgment_type == JudgmentType.PAIRWISE:
            # Use Cohen's kappa for pairwise judgments
            return self._compute_cohens_kappa(grouped), "cohens_kappa"
        else:
            # Use Fleiss' kappa for absolute/rubric judgments
            return self._compute_fleiss_kappa(grouped), "fleiss_kappa"
    
    def _compute_scores_from_judgments(
        self, judgments: List[Judgment]
    ) -> Dict[str, float]:
        """Helper to compute scores from judgments."""
        # Simple aggregation - count wins for pairwise, average for absolute
        model_scores = defaultdict(lambda: {"wins": 0, "total": 0, "scores": []})
        
        for judgment in judgments:
            if judgment.judgment_type == JudgmentType.PAIRWISE:
                model_scores[judgment.model_a_id]["total"] += 1
                model_scores[judgment.model_b_id]["total"] += 1
                
                if judgment.preference == "A":
                    model_scores[judgment.model_a_id]["wins"] += 1
                elif judgment.preference == "B":
                    model_scores[judgment.model_b_id]["wins"] += 1
                else:  # tie
                    model_scores[judgment.model_a_id]["wins"] += 0.5
                    model_scores[judgment.model_b_id]["wins"] += 0.5
                    
            elif judgment.judgment_type == JudgmentType.ABSOLUTE:
                if judgment.score is not None:
                    normalized = judgment.score / judgment.max_score if judgment.max_score else judgment.score
                    model_scores[judgment.model_id]["scores"].append(normalized)
        
        # Compute final scores
        scores = {}
        for model_id, data in model_scores.items():
            if data["scores"]:
                scores[model_id] = np.mean(data["scores"])
            elif data["total"] > 0:
                scores[model_id] = data["wins"] / data["total"]
            else:
                scores[model_id] = 0.5
        
        return scores
    
    def _group_judgments(self, judgments: List[Judgment]) -> Dict[str, List[Judgment]]:
        """Group judgments by item being judged."""
        grouped = defaultdict(list)
        
        for judgment in judgments:
            if judgment.judgment_type == JudgmentType.PAIRWISE:
                # Use sorted pair as key
                key = f"{judgment.prompt_id}:{tuple(sorted([judgment.model_a_id, judgment.model_b_id]))}"
            else:
                key = f"{judgment.prompt_id}:{judgment.model_id}"
            
            grouped[key].append(judgment)
        
        return grouped
    
    def _compute_cohens_kappa(self, grouped: Dict[str, List[Judgment]]) -> float:
        """
        Compute Cohen's kappa for pairwise judgments.
        
        For multiple judges, we compute pairwise kappas and average.
        """
        kappas = []
        
        for item_judgments in grouped.values():
            if len(item_judgments) < 2:
                continue
            
            # Extract preferences
            preferences = [j.preference for j in item_judgments]
            
            # Compute agreement
            n = len(preferences)
            observed_agreement = sum(
                1 for i in range(n) for j in range(i+1, n)
                if preferences[i] == preferences[j]
            ) / (n * (n - 1) / 2)
            
            # Expected agreement (assuming uniform distribution)
            unique_prefs = set(preferences)
            expected_agreement = 1.0 / len(unique_prefs) if unique_prefs else 0.0
            
            # Kappa
            if expected_agreement < 1.0:
                kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
                kappas.append(kappa)
        
        return float(np.mean(kappas)) if kappas else 0.0
    
    def _compute_fleiss_kappa(self, grouped: Dict[str, List[Judgment]]) -> float:
        """
        Compute Fleiss' kappa for absolute/rubric judgments.
        """
        # Build rating matrix
        all_ratings = []
        
        for item_judgments in grouped.values():
            if len(item_judgments) < 2:
                continue
            
            # Extract scores (discretized)
            ratings = []
            for judgment in item_judgments:
                if judgment.score is not None:
                    # Discretize to 5 categories
                    normalized = judgment.score / judgment.max_score if judgment.max_score else judgment.score
                    category = int(normalized * 4)  # 0-4
                    ratings.append(category)
            
            if ratings:
                all_ratings.append(ratings)
        
        if not all_ratings:
            return 0.0
        
        # Compute Fleiss' kappa
        n_items = len(all_ratings)
        n_raters = len(all_ratings[0])
        n_categories = 5
        
        # Count ratings per category per item
        rating_counts = np.zeros((n_items, n_categories))
        for i, ratings in enumerate(all_ratings):
            for rating in ratings:
                if 0 <= rating < n_categories:
                    rating_counts[i, rating] += 1
        
        # Proportion of all assignments in each category
        p_j = rating_counts.sum(axis=0) / (n_items * n_raters)
        
        # Observed agreement
        P_i = (rating_counts ** 2).sum(axis=1) - n_raters
        P_i = P_i / (n_raters * (n_raters - 1))
        P_bar = P_i.mean()
        
        # Expected agreement
        P_bar_e = (p_j ** 2).sum()
        
        # Fleiss' kappa
        if P_bar_e < 1.0:
            kappa = (P_bar - P_bar_e) / (1.0 - P_bar_e)
        else:
            kappa = 0.0
        
        return float(kappa)
