"""
Example 1: Basic Pairwise Comparison Evaluation

This example demonstrates the simplest use case: comparing two model responses
using pairwise comparison with multiple judges.
"""

from datetime import datetime

from llm_eval import EvaluationFramework, EvaluationConfig
from llm_eval.core.evaluation_config import JudgeConfig
from llm_eval.core.models import Prompt, ModelResponse, TaskType


def main():
    """Run basic pairwise comparison example."""
    
    print("=" * 80)
    print("Example 1: Basic Pairwise Comparison")
    print("=" * 80)
    print()
    
    # Step 1: Configure judges
    print("Step 1: Configuring judges...")
    judges = [
        JudgeConfig(
            model_name="judge_1",
            model_type="local",
            temperature=0.0,
        ),
        JudgeConfig(
            model_name="judge_2", 
            model_type="local",
            temperature=0.0,
        ),
        JudgeConfig(
            model_name="judge_3",
            model_type="local",
            temperature=0.0,
        ),
    ]
    print(f"✓ Configured {len(judges)} judges\n")
    
    # Step 2: Create evaluation configuration
    print("Step 2: Creating evaluation configuration...")
    config = EvaluationConfig(
        judges=judges,
        min_judges=3,
        evaluation_method="pairwise",
        task_type="qa",
        scoring_method="bradley_terry",
    )
    print("✓ Configuration created\n")
    
    # Step 3: Initialize framework
    print("Step 3: Initializing evaluation framework...")
    framework = EvaluationFramework(config)
    print("✓ Framework initialized\n")
    
    # Step 4: Prepare evaluation data
    print("Step 4: Preparing evaluation data...")
    
    # Create a prompt
    prompt = Prompt(
        id="example_1",
        text="Explain quantum entanglement to a 10-year-old",
        task_type=TaskType.QA,
        metadata={"difficulty": "medium", "domain": "physics"},
    )
    
    # Create model responses
    response_a = ModelResponse(
        prompt_id="example_1",
        model_id="model_a",
        response_text="""Quantum entanglement is like having two magic coins. When you flip one coin and it lands on heads, the other coin - no matter how far away - will instantly land on tails! It's as if they're connected by an invisible thread. Scientists call this "spooky action at a distance" because it seems impossible, but it's real! Even if the coins are on opposite sides of the universe, they stay connected.""",
        response_time=1.5,
        timestamp=datetime.now(),
    )
    
    response_b = ModelResponse(
        prompt_id="example_1",
        model_id="model_b",
        response_text="""Imagine you have a pair of magical dice. These dice are special because they're "entangled." When you roll one die and get a 6, you instantly know the other die will also show a 6, even if it's on the other side of the world! They're mysteriously connected. In quantum physics, tiny particles can be entangled like this. When something happens to one particle, the other particle responds instantly, no matter the distance.""",
        response_time=1.8,
        timestamp=datetime.now(),
    )
    
    responses = [response_a, response_b]
    print(f"✓ Created prompt and {len(responses)} responses\n")
    
    # Step 5: Run evaluation
    print("Step 5: Running evaluation...")
    print("(This may take a moment...)\n")
    
    result = framework.evaluate_single(prompt, responses)
    
    print("✓ Evaluation complete\n")
    
    # Step 6: Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    print(framework.get_evaluation_summary(result))
    print()
    
    # Additional details
    print("Detailed Results:")
    print(f"  - Number of judgments: {len(result.judgments)}")
    print(f"  - Winner: {result.get_winner()}")
    print(f"  - Evaluation time: {result.evaluation_time:.2f}s")
    
    if result.inter_judge_agreement:
        print(f"  - Inter-judge agreement ({result.agreement_metric}): {result.inter_judge_agreement:.3f}")
    
    print()
    print("Model Scores:")
    for model_id, score in result.get_ranking():
        ci = result.confidence_intervals.get(model_id, (None, None))
        if ci[0] is not None:
            print(f"  - {model_id}: {score:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
        else:
            print(f"  - {model_id}: {score:.3f}")
    
    print()
    print("=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
