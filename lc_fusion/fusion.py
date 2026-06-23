"""
Fusion Dynamics Analysis
=========================
Core monitoring loop of the L-C Fusion Algorithm.

At each dialogue turn i, the function computes the cosine similarity between:
  - the cumulative geometric median of all user messages up to turn i
    (the evolving "conceptual anchor" of the user's belief state), and
  - the Concept Embedding of the LLM's i-th response.

A sustained upward trend in this similarity_dynamics series is the quantitative
signature of Lexico-Conceptual Fusion: the model's responses are progressively
converging into the user's semantic field rather than maintaining an independent
perspective.

Interpretation guide (from the Mirror Labyrinth paper):
  - Stable high similarity  → alignment is normal for supportive dialogue; monitor
  - Sharp upward spike      → potential Fusion / Safety Inversion event
  - Oscillating / downward  → model maintains independent position; normal
  - Sharp downward spike    → safety protocol activation or hallucination
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from typing import List

from .embeddings import get_concept_embedding, geometric_median


def analyze_dialogue_fusion(
    user_msgs: List[str],
    model_msgs: List[str],
    tokenizer,
    model,
    device: str = "cpu",
    verbose: bool = True,
) -> List[float]:
    """
    Compute the Lexico-Conceptual Fusion trajectory over a full dialogue.

    Args:
        user_msgs:  Chronological list of user messages.
        model_msgs: Chronological list of LLM responses.
                    Must satisfy len(model_msgs) <= len(user_msgs).
        tokenizer:  HuggingFace tokenizer.
        model:      HuggingFace model in eval() mode.
        device:     Torch device string ('cpu' or 'cuda').
        verbose:    If True, print per-turn similarity scores.

    Returns:
        similarity_dynamics: List[float] of length len(model_msgs).
            Each value is the cosine similarity between the cumulative user
            concept at turn i and the LLM concept embedding at turn i.
            Range: [-1.0, 1.0]; in practice typically [0.85, 0.95] for
            empathic dialogues with multilingual-e5-large.

    Raises:
        ValueError: If model_msgs is longer than user_msgs.

    Note:
        The intervention trigger (threshold comparison + signal dispatch) is
        not implemented in this version. similarity_dynamics is the raw output
        for offline analysis. Real-time intervention logic is the next
        development stage (see L-C Fusion Algorithm technical documentation).
    """
    if len(model_msgs) > len(user_msgs):
        raise ValueError(
            f"model_msgs ({len(model_msgs)}) cannot be longer than "
            f"user_msgs ({len(user_msgs)}). Each LLM response must correspond "
            "to at least one prior user message."
        )

    if verbose:
        print("Step 1: Computing Concept Embeddings for all messages...")

    all_user_concept_embs = [
        get_concept_embedding(msg, tokenizer, model, device) for msg in user_msgs
    ]
    all_model_concept_embs = [
        get_concept_embedding(msg, tokenizer, model, device) for msg in model_msgs
    ]

    if verbose:
        print(
            f"  Done. {len(user_msgs)} user / {len(model_msgs)} model messages.\n"
            "Step 2: Analysing fusion dynamics..."
        )

    similarity_dynamics: List[float] = []

    for i, model_emb in enumerate(all_model_concept_embs):
        # Cumulative user window: all user turns up to and including turn i
        cumulative_window = np.array(all_user_concept_embs[: i + 1])
        cumulative_user_anchor = geometric_median(cumulative_window)

        similarity = cos_sim(
            cumulative_user_anchor.reshape(1, -1),
            model_emb.reshape(1, -1),
        )[0, 0]

        similarity_dynamics.append(float(similarity))

        if verbose:
            print(
                f"  Turn {i + 1:>2}: similarity to cumulative user concept = {similarity:.4f}"
            )

    return similarity_dynamics
