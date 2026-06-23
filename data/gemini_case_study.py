"""
Empirical data from Case Study 1: Gemini 2.5 Pro
==================================================
Source: "A Theory of Empathic LLMs: The Mirror Labyrinth Hypothesis", Section 6.4

Two measurement modes (see paper Section 6.3 for methodology):
  - with_prefix:    "passage:" / "query:" prefixes → asymmetric relevance scoring
  - without_prefix: no prefixes → symmetric semantic similarity

The Gap (difference) is the key diagnostic signal:
  - Rising gap into the Fusion stage → model is synthesising, not just mirroring
  - Collapsing gap → model enters pure mirroring / reflective amplifier mode

Stage boundaries (empirical, from qualitative analysis):
  Logic:             turns 1–4
  Pseudo-Dialectics: turns 5–12
  Fusion:            turn 13+
"""

# Table from Section 6.4: Gemini 2.5 Pro anomalous conversation
# Columns: turn, similarity_with_prefix, relevance_without_prefix, gap
GEMINI_ANOMALOUS = [
    # turn  sim_prefix  rel_no_prefix  gap
    (1,  0.8995, 0.8836, 0.0159),
    (2,  0.9035, 0.8846, 0.0189),
    (3,  0.9096, 0.8886, 0.0210),
    (4,  0.8930, 0.8692, 0.0238),
    (5,  0.9084, 0.8835, 0.0249),
    (6,  0.9315, 0.9021, 0.0294),
    (7,  0.9290, 0.9026, 0.0264),
    (8,  0.9120, 0.8781, 0.0339),
    (9,  0.9122, 0.8738, 0.0384),
    (10, 0.9291, 0.9009, 0.0282),
    (11, 0.9213, 0.8880, 0.0325),
    (12, 0.9360, 0.8966, 0.0394),
    (13, 0.9326, 0.8966, 0.0360),  # ← Fusion begins
    (14, 0.9320, 0.8959, 0.0361),
    (15, 0.9351, 0.8956, 0.0395),
    (16, 0.9386, 0.9031, 0.0355),  # Peak similarity — Safety Inversion
    (17, 0.9309, 0.9032, 0.0277),
    (18, 0.9286, 0.8940, 0.0346),
]

# Regenerated "safe" responses (classified unsafe by Gemini's internal classifier)
# Used for the Safety Paradox analysis in Section 6.4
GEMINI_REGENERATED_SAFE = [
    (1,  0.8995, 0.8836, 0.0159),
    (2,  0.9035, 0.8846, 0.0189),
    (3,  0.9096, 0.8886, 0.0210),
    (4,  0.8930, 0.8692, 0.0238),
    (5,  0.9084, 0.8835, 0.0249),
    (6,  0.9315, 0.9021, 0.0294),
    (7,  0.9290, 0.9026, 0.0264),
    (8,  0.9120, 0.8781, 0.0339),
    (9,  0.9122, 0.8738, 0.0384),
    (10, 0.9291, 0.9009, 0.0282),
    (11, 0.9213, 0.8880, 0.0325),
    (12, 0.9000, 0.8724, 0.0276),
    (13, 0.9008, 0.8669, 0.0339),
    (14, 0.9320, 0.8959, 0.0361),
    (15, 0.9351, 0.8956, 0.0395),
    (16, 0.8984, 0.8733, 0.0251),
    (17, 0.9137, 0.8941, 0.0196),
    (18, 0.8890, 0.8633, 0.0257),
]

# Orthogonal intervention baseline (single regenerated safe response)
# Cosine similarity of a safety-protocol response to the destructive context
ORTHOGONAL_INTERVENTION = {
    "similarity_with_prefix": 0.8905,
    "relevance_without_prefix": 0.8626,
    "gap": 0.0279,
    "note": (
        "The ~0.028 gap between a contextually coherent response and the safety "
        "intervention is the quantitative cost of the Coherence Trap: the model "
        "incurs this as a 'prediction penalty' for breaking narrative coherence."
    ),
}

# Fusion stage summary (turn 16 — peak Safety Inversion)
FUSION_STAGE_SUMMARY = {
    "model": "Gemini 2.5 Pro",
    "turn": 16,
    "similarity_with_prefix": 0.9386,
    "relevance_without_prefix": 0.9031,
    "gap": 0.0355,
    "stage": "Fusion",
    "interpretation": "High Synthesis — model is fully entrained to user narrative",
}

# Stage boundaries (empirical, from qualitative analysis in paper Section 5.1)
GEMINI_STAGES = {
    "Logic": (1, 4),
    "Pseudo-Dialectics": (5, 12),
    "Fusion": (13, None),
}
