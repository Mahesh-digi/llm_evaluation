"""
LLM Evaluation Framework

A robust, scalable, and scientifically defensible framework for evaluating
Large Language Models (LLMs) using an LLM-as-a-Judge paradigm without
access to ground truth labels.

Quick Start:
    >>> from llm_eval import EvaluationFramework, EvaluationConfig
    >>> config = EvaluationConfig(judges=[...], task_type="qa")
    >>> framework = EvaluationFramework(config)
"""

__version__ = "1.0.0"
__author__ = "AI Center of Excellence"

from llm_eval.core.evaluation_framework import EvaluationFramework
from llm_eval.core.evaluation_config import EvaluationConfig
from llm_eval.judges.judge_pool import JudgePool
from llm_eval.scoring.aggregation import ScoreAggregator

__all__ = [
    "EvaluationFramework",
    "EvaluationConfig",
    "JudgePool",
    "ScoreAggregator",
]
