"""
Example 3: Rubric-Based Evaluation

This example demonstrates using rubric-based evaluation for
multi-dimensional assessment of responses.
"""

from datetime import datetime

from llm_eval import EvaluationFramework, EvaluationConfig
from llm_eval.core.evaluation_config import JudgeConfig
from llm_eval.core.models import Prompt, ModelResponse, TaskType


def main():
    """Run rubric-based evaluation example."""
    
    print("=" * 80)
    print("Example 3: Rubric-Based Evaluation")
    print("=" * 80)
    print()
    
    # Configure for rubric-based evaluation
    judges = [
        JudgeConfig(model_name="judge_1", model_type="local", temperature=0.0),
        JudgeConfig(model_name="judge_2", model_type="local", temperature=0.0),
        JudgeConfig(model_name="judge_3", model_type="local", temperature=0.0),
    ]
    
    # Custom rubric for code evaluation
    custom_rubric = {
        "correctness": 10.0,
        "efficiency": 10.0,
        "readability": 10.0,
        "style": 10.0,
        "completeness": 10.0,
    }
    
    config = EvaluationConfig(
        judges=judges,
        min_judges=3,
        evaluation_method="rubric",
        task_type="code",
        scoring_method="simple_average",
        custom_rubric=custom_rubric,
    )
    
    framework = EvaluationFramework(config)
    print("✓ Framework initialized with custom rubric\n")
    
    print("Custom Rubric:")
    for criterion, max_score in custom_rubric.items():
        print(f"  - {criterion}: 0-{max_score}")
    print()
    
    # Create a code generation prompt
    prompt = Prompt(
        id="code_example",
        text="Write a Python function to find the nth Fibonacci number using dynamic programming",
        task_type=TaskType.CODE,
    )
    
    # Create responses
    responses = [
        ModelResponse(
            prompt_id="code_example",
            model_id="model_a",
            response_text="""def fibonacci(n):
    if n <= 1:
        return n
    
    # Dynamic programming approach
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]""",
            response_time=2.0,
        ),
        ModelResponse(
            prompt_id="code_example",
            model_id="model_b",
            response_text="""def fibonacci(n):
    # Base cases
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    # Use DP with space optimization
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr""",
            response_time=2.2,
        ),
        ModelResponse(
            prompt_id="code_example",
            model_id="model_c",
            response_text="""def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a""",
            response_time=1.5,
        ),
    ]
    
    print(f"✓ Created code generation task with {len(responses)} responses\n")
    
    # Run evaluation
    print("Running rubric-based evaluation...")
    print("(This may take a moment...)\n")
    
    result = framework.evaluate_single(prompt, responses)
    
    print("✓ Evaluation complete\n")
    
    # Display results
    print("=" * 80)
    print("RUBRIC-BASED RESULTS")
    print("=" * 80)
    print()
    
    print("Overall Scores:")
    for i, (model_id, score) in enumerate(result.get_ranking(), 1):
        print(f"  {i}. {model_id}: {score:.2f}/10")
    
    print()
    print("Detailed Rubric Scores:")
    print("-" * 80)
    
    # Group judgments by model
    from collections import defaultdict
    model_rubric_scores = defaultdict(lambda: defaultdict(list))
    
    for judgment in result.judgments:
        if judgment.rubric_scores:
            for criterion, score in judgment.rubric_scores.items():
                model_rubric_scores[judgment.model_id][criterion].append(score)
    
    # Display per-model rubric scores
    for model_id in sorted(model_rubric_scores.keys()):
        print(f"\n{model_id}:")
        total_score = 0
        total_max = 0
        for criterion in custom_rubric.keys():
            if criterion in model_rubric_scores[model_id]:
                scores = model_rubric_scores[model_id][criterion]
                avg_score = sum(scores) / len(scores)
                max_score = custom_rubric[criterion]
                total_score += avg_score
                total_max += max_score
                print(f"  {criterion:12s}: {avg_score:5.2f}/{max_score:.1f}")
        print(f"  {'TOTAL':12s}: {total_score:5.2f}/{total_max:.1f}")
    
    print()
    print("=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
