"""
Bias mitigation strategies for LLM-as-a-Judge evaluation.

Addresses:
- Position bias (order effects)
- Length bias (preference for longer responses)
- Self-preference bias
- Stylistic bias
"""

from typing import List, Tuple
import random
import itertools

from llm_eval.core.models import ModelResponse, Judgment
from llm_eval.core.evaluation_config import BiasConfig


class BiasMitigator:
    """
    Implements bias mitigation strategies for fair evaluation.
    
    Key strategies:
    1. Response order randomization
    2. Blind evaluation (hide model identities)
    3. Position bias correction
    4. Length normalization
    """
    
    def __init__(self, config: BiasConfig):
        """
        Initialize bias mitigator.
        
        Args:
            config: Bias mitigation configuration
        """
        self.config = config
        # Set random seed if configured (for reproducibility in testing)
        if config.random_seed is not None:
            random.seed(config.random_seed)
    
    def generate_comparison_pairs(
        self, responses: List[ModelResponse]
    ) -> List[Tuple[ModelResponse, ModelResponse]]:
        """
        Generate all pairwise comparison pairs.
        
        Args:
            responses: List of model responses
            
        Returns:
            List of (response_a, response_b) tuples
        """
        # Generate all unique pairs
        pairs = list(itertools.combinations(responses, 2))
        
        # Randomize order if configured
        if self.config.randomize_order:
            random.shuffle(pairs)
        
        return pairs
    
    def create_order_permutations(
        self,
        response_a: ModelResponse,
        response_b: ModelResponse,
        num_permutations: int = 2,
    ) -> List[Tuple[ModelResponse, ModelResponse]]:
        """
        Create order permutations to mitigate position bias.
        
        Args:
            response_a: First response
            response_b: Second response
            num_permutations: Number of permutations (typically 2)
            
        Returns:
            List of permuted pairs
        """
        if not self.config.randomize_order or num_permutations < 2:
            return [(response_a, response_b)]
        
        # Create both orderings
        permutations = [
            (response_a, response_b),
            (response_b, response_a),
        ]
        
        return permutations[:num_permutations]
    
    def aggregate_permutations(self, judgments: List[Judgment]) -> Judgment:
        """
        Aggregate judgments across order permutations.
        
        This corrects for position bias by averaging across different orderings.
        
        Args:
            judgments: List of judgments from different orderings
            
        Returns:
            Aggregated judgment
        """
        if len(judgments) == 1:
            return judgments[0]
        
        # Count preferences accounting for swapped order
        model_a_id = judgments[0].model_a_id
        model_b_id = judgments[0].model_b_id
        
        votes = {"A": 0, "B": 0, "tie": 0}
        
        for judgment in judgments:
            # Check if order was swapped
            if judgment.model_a_id == model_a_id:
                # Same order
                if judgment.preference == "A":
                    votes["A"] += 1
                elif judgment.preference == "B":
                    votes["B"] += 1
                else:
                    votes["tie"] += 1
            else:
                # Swapped order - reverse preference
                if judgment.preference == "A":
                    votes["B"] += 1
                elif judgment.preference == "B":
                    votes["A"] += 1
                else:
                    votes["tie"] += 1
        
        # Determine final preference
        max_votes = max(votes.values())
        if votes["A"] == max_votes:
            final_preference = "A"
        elif votes["B"] == max_votes:
            final_preference = "B"
        else:
            final_preference = "tie"
        
        # Create aggregated judgment
        aggregated = judgments[0]
        aggregated.preference = final_preference
        aggregated.confidence = max_votes / len(judgments)
        aggregated.metadata["permutation_aggregated"] = True
        aggregated.metadata["vote_distribution"] = votes
        
        return aggregated
    
    def apply_blind_evaluation(self, response: ModelResponse) -> ModelResponse:
        """
        Create a blinded version of a response (hide model identity).
        
        Args:
            response: Original response
            
        Returns:
            Blinded response
        """
        if not self.config.use_blind_evaluation:
            return response
        
        # Create copy with anonymized model ID
        blinded = ModelResponse(
            prompt_id=response.prompt_id,
            model_id="Model_Anonymous",
            response_text=response.response_text,
            response_time=response.response_time,
            timestamp=response.timestamp,
            metadata=response.metadata.copy(),
        )
        
        # Store original ID in metadata for later recovery
        blinded.metadata["original_model_id"] = response.model_id
        
        return blinded
    
    def normalize_length_bias(
        self, judgments: List[Judgment], responses: List[ModelResponse]
    ) -> List[Judgment]:
        """
        Normalize judgments to account for length bias.
        
        This applies a correction factor when judges show systematic preference
        for longer responses regardless of quality.
        
        Args:
            judgments: List of judgments
            responses: List of responses
            
        Returns:
            Corrected judgments
        """
        if not self.config.length_normalization:
            return judgments
        
        # Build response length map
        response_lengths = {
            r.model_id: len(r.response_text) for r in responses
        }
        
        # Detect length bias in pairwise judgments
        length_bias_score = self._detect_length_bias(judgments, response_lengths)
        
        # If significant length bias detected, apply correction
        if abs(length_bias_score) > 0.2:  # Threshold for correction
            corrected = []
            for judgment in judgments:
                corrected_judgment = self._apply_length_correction(
                    judgment, response_lengths, length_bias_score
                )
                corrected.append(corrected_judgment)
            return corrected
        
        return judgments
    
    def _detect_length_bias(
        self, judgments: List[Judgment], response_lengths: dict
    ) -> float:
        """
        Detect if judges systematically prefer longer responses.
        
        Returns:
            Bias score: positive if preferring longer, negative if shorter
        """
        bias_votes = []
        
        for judgment in judgments:
            if judgment.judgment_type.value != "pairwise":
                continue
            
            len_a = response_lengths.get(judgment.model_a_id, 0)
            len_b = response_lengths.get(judgment.model_b_id, 0)
            
            if len_a == len_b:
                continue
            
            longer_is_a = len_a > len_b
            
            if judgment.preference == "A" and longer_is_a:
                bias_votes.append(1)
            elif judgment.preference == "B" and not longer_is_a:
                bias_votes.append(1)
            elif judgment.preference == "tie":
                bias_votes.append(0)
            else:
                bias_votes.append(-1)
        
        if not bias_votes:
            return 0.0
        
        return sum(bias_votes) / len(bias_votes)
    
    def _apply_length_correction(
        self, judgment: Judgment, response_lengths: dict, bias_score: float
    ) -> Judgment:
        """Apply length bias correction to a single judgment."""
        if judgment.judgment_type.value != "pairwise":
            return judgment
        
        len_a = response_lengths.get(judgment.model_a_id, 0)
        len_b = response_lengths.get(judgment.model_b_id, 0)
        
        if len_a == len_b:
            return judgment
        
        # Adjust confidence based on length difference and bias
        length_diff_ratio = abs(len_a - len_b) / max(len_a, len_b)
        correction_factor = bias_score * length_diff_ratio
        
        # Apply correction
        corrected = judgment
        if corrected.confidence:
            corrected.confidence = max(0.0, min(1.0, corrected.confidence - abs(correction_factor)))
        
        corrected.metadata["length_bias_correction_applied"] = True
        corrected.metadata["correction_factor"] = correction_factor
        
        return corrected
    
    def detect_self_preference_bias(
        self, judgments: List[Judgment], judge_model_family: str
    ) -> float:
        """
        Detect if a judge shows preference for models from the same family.
        
        Args:
            judgments: List of judgments
            judge_model_family: Family/provider of the judge model (e.g., "openai", "anthropic")
            
        Returns:
            Self-preference score (higher = more bias)
        """
        same_family_wins = 0
        different_family_wins = 0
        
        for judgment in judgments:
            if judgment.judgment_type.value != "pairwise":
                continue
            
            # Assume model IDs contain family information
            # This is a simplified check
            model_a_family = self._extract_model_family(judgment.model_a_id)
            model_b_family = self._extract_model_family(judgment.model_b_id)
            
            if judgment.preference == "tie":
                continue
            
            winner_id = judgment.model_a_id if judgment.preference == "A" else judgment.model_b_id
            winner_family = self._extract_model_family(winner_id)
            
            if winner_family == judge_model_family:
                same_family_wins += 1
            else:
                different_family_wins += 1
        
        total = same_family_wins + different_family_wins
        if total == 0:
            return 0.0
        
        # Return bias score (0.5 = no bias, >0.5 = self-preference)
        return same_family_wins / total
    
    def _extract_model_family(self, model_id: str) -> str:
        """Extract model family from model ID."""
        model_id_lower = model_id.lower()
        
        if "gpt" in model_id_lower or "openai" in model_id_lower:
            return "openai"
        elif "claude" in model_id_lower or "anthropic" in model_id_lower:
            return "anthropic"
        elif "llama" in model_id_lower or "meta" in model_id_lower:
            return "meta"
        elif "gemini" in model_id_lower or "google" in model_id_lower:
            return "google"
        else:
            return "unknown"
