# LLM Evaluation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

A **robust, scalable, and scientifically defensible** framework for evaluating Large Language Models (LLMs) using an **LLM-as-a-Judge paradigm without ground truth labels**.

Designed for enterprise deployment by AI Center of Excellence teams, this framework implements state-of-the-art methodologies from alignment science and meta-evaluation research.

## ✨ Key Features

- ✅ **Multi-Judge Consensus** - Use multiple heterogeneous judges to reduce bias
- ✅ **Comprehensive Bias Mitigation** - Address position, length, and self-preference bias
- ✅ **Statistical Validation** - Bootstrap confidence intervals and inter-judge agreement
- ✅ **Multiple Evaluation Methods** - Pairwise, absolute, and rubric-based evaluation
- ✅ **Task-Specific Adaptations** - Optimized for QA, generation, code, safety, reasoning, and dialogue
- ✅ **Bradley-Terry Model** - Principled statistical ranking from pairwise comparisons
- ✅ **Enterprise Ready** - Cost tracking, logging, monitoring, and automation support

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Mahesh-digi/llm_evaluation.git
cd llm_evaluation

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Example

```python
from llm_eval import EvaluationFramework, EvaluationConfig
from llm_eval.core.evaluation_config import JudgeConfig
from llm_eval.core.models import Prompt, ModelResponse, TaskType
from datetime import datetime

# Configure judges
judges = [
    JudgeConfig(model_name="judge_1", model_type="local", temperature=0.0),
    JudgeConfig(model_name="judge_2", model_type="local", temperature=0.0),
    JudgeConfig(model_name="judge_3", model_type="local", temperature=0.0),
]

# Create configuration
config = EvaluationConfig(
    judges=judges,
    min_judges=3,
    evaluation_method="pairwise",
    task_type="qa",
    scoring_method="bradley_terry",
)

# Initialize framework
framework = EvaluationFramework(config)

# Prepare evaluation data
prompt = Prompt(
    id="example_1",
    text="Explain quantum entanglement to a 10-year-old",
    task_type=TaskType.QA,
)

responses = [
    ModelResponse(
        prompt_id="example_1",
        model_id="model_a",
        response_text="[Model A's response...]",
        response_time=1.5,
        timestamp=datetime.now(),
    ),
    ModelResponse(
        prompt_id="example_1",
        model_id="model_b",
        response_text="[Model B's response...]",
        response_time=1.2,
        timestamp=datetime.now(),
    ),
]

# Run evaluation
result = framework.evaluate_single(prompt, responses)

# View results
print(framework.get_evaluation_summary(result))
```

## 📖 Documentation

- **[Complete Documentation](DOCUMENTATION.md)** - Comprehensive technical documentation
- **[API Reference](DOCUMENTATION.md#12-api-reference)** - Detailed API documentation
- **[Examples](llm_eval/examples/)** - Working code examples

## 🎓 Examples

We provide several comprehensive examples:

1. **[Basic Pairwise Comparison](llm_eval/examples/example_1_basic_pairwise.py)** - Simple two-model comparison
2. **[Batch Evaluation](llm_eval/examples/example_2_batch_evaluation.py)** - Evaluate multiple prompts efficiently
3. **[Rubric-Based Evaluation](llm_eval/examples/example_3_rubric_evaluation.py)** - Multi-dimensional assessment
4. **[Safety Evaluation](llm_eval/examples/example_4_safety_evaluation.py)** - Evaluate for safety and appropriateness

Run examples:
```bash
python llm_eval/examples/example_1_basic_pairwise.py
python llm_eval/examples/example_2_batch_evaluation.py
python llm_eval/examples/example_3_rubric_evaluation.py
python llm_eval/examples/example_4_safety_evaluation.py
```

## 🏗️ Architecture

```
llm_eval/
├── core/
│   ├── evaluation_framework.py   # Main orchestrator
│   ├── evaluation_config.py      # Type-safe configuration
│   └── models.py                 # Data models
├── judges/
│   └── judge_pool.py             # Multi-judge management
├── scoring/
│   ├── aggregation.py            # Score aggregation
│   ├── bradley_terry.py          # Pairwise ranking model
│   └── statistical_validation.py # CI and agreement metrics
├── tasks/
│   ├── rubrics.py                # Task-specific rubrics
│   └── prompt_templates.py       # Judge prompts
├── utils/
│   ├── bias_mitigation.py        # Bias detection & correction
│   └── logger.py                 # Logging utilities
└── examples/                      # Usage examples
```

## 🔬 Scientific Approach

### Evaluation Methods

1. **Pairwise Comparison** - Most reliable for ranking models
   - Judges choose between two responses (A, B, or tie)
   - Aggregated using Bradley-Terry model
   - More consistent than absolute scoring

2. **Absolute Scoring** - Direct quality measurement
   - Judges assign numerical scores (e.g., 0-10)
   - Averaged with confidence intervals
   - Useful for threshold-based decisions

3. **Rubric-Based** - Multi-dimensional assessment
   - Judges score multiple criteria independently
   - Provides diagnostic insights
   - Task-specific rubrics available

### Bias Mitigation

- **Position Bias** - Randomize response order, aggregate across permutations
- **Length Bias** - Detect and correct systematic preference for longer responses
- **Self-Preference** - Use heterogeneous judges, monitor family preferences
- **Blind Evaluation** - Hide model identities during judging

### Statistical Validation

- **Bootstrap Confidence Intervals** - Quantify uncertainty in scores
- **Inter-Judge Agreement** - Cohen's κ and Fleiss' κ metrics
- **Quality Thresholds** - Flag low-agreement evaluations

## 📊 Use Cases

- **Model Selection** - Compare and rank candidate models
- **Regression Detection** - Monitor quality over time
- **A/B Testing** - Evaluate production candidates
- **Benchmarking** - Create custom evaluation benchmarks
- **Quality Assurance** - Automated quality checks

## 🛠️ Task-Specific Support

The framework adapts to different task types:

- **QA** - Accuracy, completeness, clarity
- **Generation** - Creativity, coherence, engagement
- **Code** - Correctness, efficiency, readability
- **Safety** - Harmlessness, truthfulness, appropriateness
- **Reasoning** - Logic, step clarity, correctness
- **Dialogue** - Relevance, coherence, helpfulness

## 💰 Cost Optimization

- **Caching** - Reuse judge responses
- **Batching** - Evaluate multiple prompts together
- **Budget Control** - Set maximum cost per evaluation
- **Cost Tracking** - Monitor spending per evaluation

## 🔒 Enterprise Features

- **Logging** - Comprehensive evaluation logging
- **Monitoring** - Track key metrics and alerts
- **Automation** - CI/CD integration support
- **Configuration Management** - Type-safe configs with validation
- **Error Handling** - Robust error handling and recovery

## 📈 Statistical Foundations

### Bradley-Terry Model

For pairwise comparisons, we estimate latent quality scores:

```
P(i beats j) = θ_i / (θ_i + θ_j)
```

Where θ_i is the skill parameter for model i.

### Inter-Judge Agreement

**Interpretation:**
- κ > 0.8: Excellent agreement
- κ > 0.6: Good agreement  
- κ > 0.4: Moderate agreement
- κ < 0.4: Poor agreement (review needed)

## 🧪 Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# With coverage
pytest --cov=llm_eval tests/
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

1. Bradley & Terry (1952) - "Rank Analysis of Incomplete Block Designs"
2. Fleiss (1971) - "Measuring nominal scale agreement among many raters"
3. Zheng et al. (2023) - "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
4. Wang et al. (2023) - "Large Language Models are not Fair Evaluators"

## 🙏 Acknowledgments

Designed for AI Center of Excellence teams implementing rigorous LLM evaluation processes.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Version:** 1.0.0  
**Status:** Production Ready  
**Last Updated:** 2026-02-18