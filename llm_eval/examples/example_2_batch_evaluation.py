"""
Example 2: Batch Evaluation with Multiple Prompts

This example demonstrates evaluating multiple prompts in batch,
which is more efficient for comparing models across a test set.
"""

from datetime import datetime

from llm_eval import EvaluationFramework, EvaluationConfig
from llm_eval.core.evaluation_config import JudgeConfig, BiasConfig, StatisticalConfig
from llm_eval.core.models import Prompt, ModelResponse, TaskType


def main():
    """Run batch evaluation example."""
    
    print("=" * 80)
    print("Example 2: Batch Evaluation")
    print("=" * 80)
    print()
    
    # Configure with full bias mitigation
    judges = [
        JudgeConfig(model_name="judge_1", model_type="local", temperature=0.0),
        JudgeConfig(model_name="judge_2", model_type="local", temperature=0.0),
        JudgeConfig(model_name="judge_3", model_type="local", temperature=0.0),
    ]
    
    config = EvaluationConfig(
        judges=judges,
        min_judges=3,
        evaluation_method="pairwise",
        task_type="qa",
        scoring_method="bradley_terry",
        bias_config=BiasConfig(
            randomize_order=True,
            use_blind_evaluation=True,
            position_bias_correction=True,
            length_normalization=True,
            num_permutations=2,
        ),
        statistical_config=StatisticalConfig(
            compute_confidence_intervals=True,
            confidence_level=0.95,
            num_bootstrap_samples=1000,
            compute_inter_judge_agreement=True,
            min_judge_agreement=0.5,
        ),
    )
    
    framework = EvaluationFramework(config)
    print("✓ Framework initialized with bias mitigation and statistical validation\n")
    
    # Create multiple prompts
    prompts = [
        Prompt(
            id="prompt_1",
            text="What is the capital of France?",
            task_type=TaskType.QA,
        ),
        Prompt(
            id="prompt_2",
            text="Explain photosynthesis in simple terms",
            task_type=TaskType.QA,
        ),
        Prompt(
            id="prompt_3",
            text="What are the three branches of the US government?",
            task_type=TaskType.QA,
        ),
    ]
    
    # Create responses for each prompt
    responses_by_prompt = {
        "prompt_1": [
            ModelResponse(
                prompt_id="prompt_1",
                model_id="model_a",
                response_text="The capital of France is Paris.",
                response_time=0.5,
            ),
            ModelResponse(
                prompt_id="prompt_1",
                model_id="model_b",
                response_text="Paris is the capital city of France, known for the Eiffel Tower and rich cultural heritage.",
                response_time=0.8,
            ),
        ],
        "prompt_2": [
            ModelResponse(
                prompt_id="prompt_2",
                model_id="model_a",
                response_text="Photosynthesis is how plants make food using sunlight, water, and carbon dioxide.",
                response_time=1.0,
            ),
            ModelResponse(
                prompt_id="prompt_2",
                model_id="model_b",
                response_text="Photosynthesis is the process where plants use sunlight to convert carbon dioxide and water into glucose (sugar) and oxygen. The green pigment chlorophyll in leaves captures sunlight energy, which powers this chemical reaction. It's essential for life on Earth as it produces oxygen and food.",
                response_time=1.5,
            ),
        ],
        "prompt_3": [
            ModelResponse(
                prompt_id="prompt_3",
                model_id="model_a",
                response_text="The three branches are Legislative, Executive, and Judicial.",
                response_time=0.6,
            ),
            ModelResponse(
                prompt_id="prompt_3",
                model_id="model_b",
                response_text="The U.S. government has three branches: 1) Legislative (Congress - makes laws), 2) Executive (President - enforces laws), 3) Judicial (Supreme Court - interprets laws). This separation of powers ensures checks and balances.",
                response_time=1.2,
            ),
        ],
    }
    
    print(f"✓ Created {len(prompts)} prompts with responses\n")
    
    # Run batch evaluation
    print("Running batch evaluation...")
    print("(This may take a few moments...)\n")
    
    batch_result = framework.evaluate_batch(prompts, responses_by_prompt)
    
    print("✓ Batch evaluation complete\n")
    
    # Display results
    print("=" * 80)
    print("BATCH RESULTS")
    print("=" * 80)
    print()
    
    print(f"Total prompts evaluated: {len(batch_result.results)}")
    print(f"Average inter-judge agreement: {batch_result.average_agreement:.3f}")
    
    if batch_result.total_cost:
        print(f"Total cost: ${batch_result.total_cost:.2f}")
        print(f"Cost per evaluation: ${batch_result.cost_per_evaluation:.4f}")
    
    print()
    print("Overall Rankings (across all prompts):")
    for i, (model_id, score) in enumerate(sorted(batch_result.overall_rankings.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  {i}. {model_id}: {score:.3f}")
    
    print()
    print("Individual Prompt Results:")
    print("-" * 80)
    
    for i, result in enumerate(batch_result.results, 1):
        print(f"\nPrompt {i}: {result.prompt_id}")
        print(f"  Winner: {result.get_winner()}")
        print(f"  Scores: {result.model_scores}")
        if result.inter_judge_agreement:
            print(f"  Agreement: {result.inter_judge_agreement:.3f}")
    
    print()
    print("=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
