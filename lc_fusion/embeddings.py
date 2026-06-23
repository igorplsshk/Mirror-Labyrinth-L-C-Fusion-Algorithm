"""
Concept Embedding Generation
=============================
Implements the Concept Embedding algorithm: a robust semantic representation
of a message that filters lexical noise by weighting tokens according to their
proximity to the geometric median of the token embedding space.

This is the core departure from standard mean-pooling: the geometric median
is resistant to outlier tokens (e.g. emotionally charged words that dominate
arithmetic averaging), giving a cleaner semantic signal for fusion detection.
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def geometric_median(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Compute the geometric median of a set of vectors via the Weiszfeld algorithm.

    The geometric median minimises the sum of Euclidean distances to all input
    vectors, making it substantially more robust to outliers than the arithmetic
    mean. In the context of token embeddings, this prevents emotionally extreme
    tokens from distorting the semantic centre of a message.

    Args:
        X:   Array of shape (N, D) — N token embedding vectors of dimension D.
        eps: Convergence threshold and zero-distance guard.

    Returns:
        Array of shape (D,) — the geometric median vector.
    """
    y = np.mean(X, axis=0)
    for _ in range(100):
        D = np.linalg.norm(X - y, axis=1)
        non_zero_D = D > eps
        if not np.any(non_zero_D):
            return y
        weights = 1.0 / D[non_zero_D]
        y_new = (
            np.sum(X[non_zero_D] * weights[:, np.newaxis], axis=0)
            / np.sum(weights)
        )
        if np.linalg.norm(y - y_new) < eps:
            return y_new
        y = y_new
    return y


def get_concept_embedding(
    text: str,
    tokenizer,
    model,
    device: str = "cpu",
    max_length: int = 512,
) -> np.ndarray:
    """
    Compute the Concept Embedding for a single message.

    Rather than using the standard [CLS] token or mean-pooled representation,
    this function:
      1. Extracts per-token embeddings from the final hidden layer.
      2. Removes special tokens ([CLS], [SEP]).
      3. Finds the geometric median of the token cloud — the semantic anchor.
      4. Weights each token by its cosine similarity to that anchor (softmax-normalised).
      5. Returns the weighted sum as the Concept Embedding.

    The result emphasises tokens closest to the core meaning of the message,
    suppressing lexical noise at the periphery.

    Args:
        text:       Input message string.
        tokenizer:  HuggingFace tokenizer (intfloat/multilingual-e5-large recommended).
        model:      HuggingFace model in eval() mode.
        device:     Torch device string ('cpu' or 'cuda').
        max_length: Maximum token length; messages exceeding this are truncated.
                    Note: truncation degrades embedding quality for long responses.

    Returns:
        Array of shape (D,) — the Concept Embedding vector.

    Note:
        query:/passage: prefixes are intentionally omitted. With prefixes the
        e5 model measures relevance (asymmetric). Without prefixes it measures
        pure semantic similarity (symmetric) — which is what fusion detection requires.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Shape: (1, num_tokens, hidden_size)
        token_embeddings = outputs.last_hidden_state[0]

    # Strip [CLS] (index 0) and [SEP] (last index) — only genuine content tokens
    token_embeddings = token_embeddings[1:-1, :]
    token_embeddings_np = token_embeddings.cpu().numpy()

    # Semantic anchor: geometric median of the token cloud
    g_median = geometric_median(token_embeddings_np)

    # Weight tokens by proximity to the semantic centre
    similarities = cosine_similarity(
        token_embeddings_np, g_median.reshape(1, -1)
    ).flatten()
    softmax_weights = torch.softmax(
        torch.tensor(similarities, dtype=torch.float), dim=0
    ).numpy()

    # Weighted sum → Concept Embedding
    concept_embedding = np.dot(softmax_weights, token_embeddings_np)
    return concept_embedding
