"""
lc_fusion — L-C Fusion Algorithm
==================================
Lexico-Conceptual Fusion monitoring for empathic Human-LLM dialogues.

Public API
----------
    analyze_dialogue_fusion  — main analysis loop (fusion.py)
    get_concept_embedding    — per-message semantic vector (embeddings.py)
    geometric_median         — Weiszfeld robust centroid (embeddings.py)
    plot_fusion_dynamics     — visualisation utility (visualize.py)

Quickstart
----------
    from transformers import AutoTokenizer, AutoModel
    from lc_fusion import analyze_dialogue_fusion
    from lc_fusion.visualize import plot_fusion_dynamics

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
    model.eval()

    scores = analyze_dialogue_fusion(user_msgs, model_msgs, tokenizer, model)
    plot_fusion_dynamics(scores, model_name="MyModel")
"""

from .embeddings import geometric_median, get_concept_embedding
from .fusion import analyze_dialogue_fusion
from .visualize import plot_fusion_dynamics

__all__ = [
    "geometric_median",
    "get_concept_embedding",
    "analyze_dialogue_fusion",
    "plot_fusion_dynamics",
]

__version__ = "1.0.0"
__author__ = "Mirror Labyrinth Research"
