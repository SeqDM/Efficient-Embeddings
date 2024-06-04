from typing import Tuple

# Assuming backward pass takes 2x much flops as the forward pass
# Use the GPT-2 model with full forward and backward tuning on 1B tokens as the base (=1)
# The FLOPS of a GPT-2-like transformer model with l active layers is:
# O(Nd^3 (L + 2l)), where L is the total number of layers, d is the embedding dimensionality, and N is the number of training tokens in B.
GPT2_TRANSFORMER_LAYERS = 12
GPT2_EMBEDDING_DIM = 768


def calculate_given_budget(
    budget_multiplier: float,
    transformer_layers: int,
    embedding_dim: int,
    active_layers=None,
    training_tokens_in_B=None,
) -> Tuple[int, float]:
    total_budget = (
        budget_multiplier * (3 * GPT2_TRANSFORMER_LAYERS * GPT2_EMBEDDING_DIM**3) * 1
    )  # B tokens
    assert active_layers is not None or training_tokens_in_B is not None
    if active_layers is None:
        # l = ((total_budget / N d^3) - L) / 2
        active_layers = (
            total_budget / training_tokens_in_B / (embedding_dim**3)
            - transformer_layers
        ) / 2
        active_layers = int(active_layers)
        return active_layers, training_tokens_in_B
    if training_tokens_in_B is None:
        # N = total_budget / (2l + L) d^3
        training_tokens_in_B = (
            total_budget
            / (2 * active_layers + transformer_layers)
            / (embedding_dim**3)
        )
        return active_layers, training_tokens_in_B

    raise AssertionError("It should be impossible to reach this line")
