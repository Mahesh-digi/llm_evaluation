"""
Main evaluation framework orchestrator.

This module provides the central EvaluationFramework class that coordinates
all components of the LLM evaluation system.
"""

from typing import List, Dict, Optional, Any
import time
from datetime import datetime

from llm_eval.core.evaluation_config import EvaluationConfig
from llm_eval.core.models import (
    Prompt,
    ModelResponse,
    Judgment,
    EvaluationResult,
    BatchEvaluationResult,
    JudgmentType,
)
from llm_eval.judges.judge_pool import JudgePool
from llm_eval.scoring.aggregation import ScoreAggregator
from llm_eval.scoring.bradley_terry import BradleyTerryModel
from llm_eval.scoring.statistical_validation import StatisticalValidator
from llm_eval.utils.bias_mitigation import BiasMitigator
from llm_eval.utils.logger import EvaluationLogger


class EvaluationFramework:
    """
    Main framework for conducting LLM evaluations using LLM-as-a-Judge paradigm.
    
    This framework provides:
    - Multi-judge consensus
    - Bias mitigation
    - Statistical validation
    - Multiple evaluation methods (pairwise, absolute, rubric-based)
    - Task-specific adaptations
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the evaluation framework.
        
        Args:
            config: EvaluationConfig object with all settings
        """
        self.config = config
        self.logger = EvaluationLogger(config.metadata.get("log_level", "INFO"))
        
        # Initialize components
        self.judge_pool = JudgePool(config.judges)
        self.score_aggregator = ScoreAggregator(config.scoring_method)
        self.bias_mitigator = BiasMitigator(config.bias_config)
        self.statistical_validator = StatisticalValidator(config.statistical_config)
        
        # Bradley-Terry model for pairwise comparisons
        self.bradley_terry = BradleyTerryModel() if config.evaluation_method == "pairwise" else None
        
        self.logger.info(f"Initialized EvaluationFramework with {len(config.judges)} judges")
        
    def evaluate_single(
        self,
        prompt: Prompt,
        responses: List[ModelResponse],
    ) -> EvaluationResult:
        """
        Evaluate responses to a single prompt.
        
        Args:
            prompt: The prompt that was evaluated
            responses: List of model responses to evaluate
            
        Returns:
            EvaluationResult with aggregated scores and judgments
        """
        start_time = time.time()
        self.logger.info(f"Starting evaluation for prompt: {prompt.id}")
        
        # Validate inputs
        if len(responses) < 2 and self.config.evaluation_method == "pairwise":
            raise ValueError("Pairwise comparison requires at least 2 responses")
        
        # Collect judgments from all judges
        all_judgments = []
        
        if self.config.evaluation_method == "pairwise":
            all_judgments = self._collect_pairwise_judgments(prompt, responses)
        elif self.config.evaluation_method == "absolute":
            all_judgments = self._collect_absolute_judgments(prompt, responses)
        elif self.config.evaluation_method == "rubric":
            all_judgments = self._collect_rubric_judgments(prompt, responses)
        elif self.config.evaluation_method == "hybrid":
            # Combine multiple methods
            all_judgments.extend(self._collect_pairwise_judgments(prompt, responses))
            all_judgments.extend(self._collect_absolute_judgments(prompt, responses))
        
        # Aggregate scores
        model_scores = self.score_aggregator.aggregate(all_judgments, responses)
        
        # Compute confidence intervals if configured
        confidence_intervals = {}
        if self.config.statistical_config.compute_confidence_intervals:
            confidence_intervals = self.statistical_validator.compute_confidence_intervals(
                all_judgments, model_scores
            )
        
        # Compute inter-judge agreement
        inter_judge_agreement = None
        agreement_metric = None
        if self.config.statistical_config.compute_inter_judge_agreement:
            inter_judge_agreement, agreement_metric = (
                self.statistical_validator.compute_inter_judge_agreement(all_judgments)
            )
            
            # Check if agreement meets threshold
            if inter_judge_agreement < self.config.statistical_config.min_judge_agreement:
                self.logger.warning(
                    f"Inter-judge agreement ({inter_judge_agreement:.3f}) below threshold "
                    f"({self.config.statistical_config.min_judge_agreement})"
                )
        
        # Calculate total cost
        total_cost = sum(j.metadata.get("cost", 0.0) for j in all_judgments)
        
        evaluation_time = time.time() - start_time
        
        result = EvaluationResult(
            prompt_id=prompt.id,
            judgments=all_judgments,
            model_scores=model_scores,
            confidence_intervals=confidence_intervals,
            inter_judge_agreement=inter_judge_agreement,
            agreement_metric=agreement_metric,
            evaluation_time=evaluation_time,
            total_cost=total_cost,
            timestamp=datetime.now(),
        )
        
        self.logger.info(
            f"Completed evaluation for prompt {prompt.id} in {evaluation_time:.2f}s. "
            f"Winner: {result.get_winner()}"
        )
        
        return result
    
    def evaluate_batch(
        self,
        prompts: List[Prompt],
        responses_by_prompt: Dict[str, List[ModelResponse]],
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple prompts in batch.
        
        Args:
            prompts: List of prompts
            responses_by_prompt: Dict mapping prompt_id to list of responses
            
        Returns:
            BatchEvaluationResult with aggregated results
        """
        self.logger.info(f"Starting batch evaluation of {len(prompts)} prompts")
        
        results = []
        total_cost = 0.0
        
        for prompt in prompts:
            if prompt.id not in responses_by_prompt:
                self.logger.warning(f"No responses found for prompt: {prompt.id}")
                continue
                
            responses = responses_by_prompt[prompt.id]
            result = self.evaluate_single(prompt, responses)
            results.append(result)
            total_cost += result.total_cost or 0.0
        
        # Create batch result
        batch_result = BatchEvaluationResult(
            results=results,
            total_cost=total_cost,
            cost_per_evaluation=total_cost / len(results) if results else 0.0,
            timestamp=datetime.now(),
        )
        
        # Compute overall rankings
        batch_result.compute_overall_rankings()
        
        # Compute average agreement
        agreements = [r.inter_judge_agreement for r in results if r.inter_judge_agreement]
        if agreements:
            batch_result.average_agreement = sum(agreements) / len(agreements)
        
        self.logger.info(
            f"Completed batch evaluation. Total cost: ${total_cost:.2f}, "
            f"Avg agreement: {batch_result.average_agreement:.3f if batch_result.average_agreement else 'N/A'}"
        )
        
        return batch_result
    
    def _collect_pairwise_judgments(
        self, prompt: Prompt, responses: List[ModelResponse]
    ) -> List[Judgment]:
        """Collect pairwise comparison judgments from all judges."""
        judgments = []
        
        # Generate all pairs
        pairs = self.bias_mitigator.generate_comparison_pairs(responses)
        
        for model_a, model_b in pairs:
            # Get judgments from all judges with order randomization
            for judge in self.judge_pool.get_active_judges():
                # Randomize order if configured
                if self.config.bias_config.randomize_order:
                    permutations = self.bias_mitigator.create_order_permutations(
                        model_a, model_b, self.config.bias_config.num_permutations
                    )
                    
                    judge_results = []
                    for perm_a, perm_b in permutations:
                        judgment = judge.judge_pairwise(prompt, perm_a, perm_b)
                        judge_results.append(judgment)
                    
                    # Aggregate across permutations
                    final_judgment = self.bias_mitigator.aggregate_permutations(judge_results)
                    judgments.append(final_judgment)
                else:
                    judgment = judge.judge_pairwise(prompt, model_a, model_b)
                    judgments.append(judgment)
        
        return judgments
    
    def _collect_absolute_judgments(
        self, prompt: Prompt, responses: List[ModelResponse]
    ) -> List[Judgment]:
        """Collect absolute score judgments from all judges."""
        judgments = []
        
        for response in responses:
            for judge in self.judge_pool.get_active_judges():
                judgment = judge.judge_absolute(prompt, response)
                judgments.append(judgment)
        
        return judgments
    
    def _collect_rubric_judgments(
        self, prompt: Prompt, responses: List[ModelResponse]
    ) -> List[Judgment]:
        """Collect rubric-based judgments from all judges."""
        judgments = []
        
        rubric = self.config.custom_rubric or self._get_default_rubric(prompt.task_type)
        
        for response in responses:
            for judge in self.judge_pool.get_active_judges():
                judgment = judge.judge_rubric(prompt, response, rubric)
                judgments.append(judgment)
        
        return judgments
    
    def _get_default_rubric(self, task_type: str) -> Dict[str, Any]:
        """Get default rubric for a task type."""
        # Import task-specific rubrics
        from llm_eval.tasks.rubrics import get_rubric_for_task
        return get_rubric_for_task(task_type)
    
    def get_evaluation_summary(self, result: EvaluationResult) -> str:
        """
        Generate a human-readable summary of evaluation results.
        
        Args:
            result: EvaluationResult to summarize
            
        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("=" * 80)
        summary.append(f"EVALUATION SUMMARY - Prompt: {result.prompt_id}")
        summary.append("=" * 80)
        summary.append("")
        
        # Rankings
        summary.append("MODEL RANKINGS:")
        for i, (model_id, score) in enumerate(result.get_ranking(), 1):
            ci = result.confidence_intervals.get(model_id, (None, None))
            ci_str = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else ""
            summary.append(f"  {i}. {model_id}: {score:.3f}{ci_str}")
        
        summary.append("")
        
        # Statistics
        if result.inter_judge_agreement is not None:
            summary.append(f"Inter-Judge Agreement ({result.agreement_metric}): {result.inter_judge_agreement:.3f}")
        
        summary.append(f"Evaluation Time: {result.evaluation_time:.2f}s")
        if result.total_cost:
            summary.append(f"Total Cost: ${result.total_cost:.4f}")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)
