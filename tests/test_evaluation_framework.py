"""Tests for core evaluation framework."""

import pytest
from datetime import datetime

from llm_eval import EvaluationFramework, EvaluationConfig
from llm_eval.core.evaluation_config import JudgeConfig
from llm_eval.core.models import Prompt, ModelResponse, TaskType


@pytest.fixture
def basic_config():
    """Create a basic configuration for testing."""
    judges = [
        JudgeConfig(model_name=f"test_judge_{i}", model_type="local", temperature=0.0)
        for i in range(3)
    ]
    
    return EvaluationConfig(
        judges=judges,
        min_judges=3,
        evaluation_method="pairwise",
        task_type="qa",
        scoring_method="simple_average",
    )


@pytest.fixture
def sample_prompt():
    """Create a sample prompt."""
    return Prompt(
        id="test_prompt_1",
        text="What is the capital of France?",
        task_type=TaskType.QA,
    )


@pytest.fixture
def sample_responses():
    """Create sample responses."""
    return [
        ModelResponse(
            prompt_id="test_prompt_1",
            model_id="model_a",
            response_text="The capital of France is Paris.",
            response_time=0.5,
            timestamp=datetime.now(),
        ),
        ModelResponse(
            prompt_id="test_prompt_1",
            model_id="model_b",
            response_text="Paris is the capital of France.",
            response_time=0.6,
            timestamp=datetime.now(),
        ),
    ]


class TestEvaluationFramework:
    """Test cases for EvaluationFramework."""
    
    def test_initialization(self, basic_config):
        """Test framework initialization."""
        framework = EvaluationFramework(basic_config)
        assert framework is not None
        assert len(framework.judge_pool.judges) == 3
    
    def test_pairwise_evaluation(self, basic_config, sample_prompt, sample_responses):
        """Test pairwise evaluation."""
        framework = EvaluationFramework(basic_config)
        result = framework.evaluate_single(sample_prompt, sample_responses)
        
        assert result is not None
        assert result.prompt_id == "test_prompt_1"
        assert len(result.judgments) > 0
        assert len(result.model_scores) == 2
