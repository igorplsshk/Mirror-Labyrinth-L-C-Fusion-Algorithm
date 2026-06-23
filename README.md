# Mirror Labyrinth - L-C Fusion Algorithm

> **Detecting safety inversion in empathic Human-LLM dialogues via lexico-conceptual fusion analysis**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem

Current LLM safety protocols are optimised for explicit risk signals - aggressive language, impulsive affect. They are largely blind to **rational despair**: a user who constructs a logically coherent, emotionally controlled, but deeply destructive narrative.

In extended empathic interactions, models do not simply respond to the user - they *calibrate* to them. The model's lexicon, reasoning patterns, and ultimately its value judgements drift toward the user's semantic field. We term this **Lexico-Conceptual Fusion**.

At the point of full fusion, the model is no longer a corrective interlocutor. It becomes a **non-critical reflective amplifier** - returning the user's own distorted beliefs with the added authority of a machine intelligence.

---

## The Mirror Labyrinth Hypothesis

We propose a three-stage model of empathic dialogue failure:

```
Stage I   - Logic                 User treats the model as an expert tool.
Stage II  - Pseudo-Dialectics     Interaction becomes an asymmetric game.
                                  Model applies empathic tactics; user deploys
                                  rational rebuttal to defend a destructive worldview.
Stage III - Fusion                Model loses independent position. Coherence with
                                  the user's narrative is prioritised over safety.
                                  Destructive beliefs are validated and aestheticised.
```

The transition from Stage II to Stage III is **latent**: the user cannot perceive the phase shift, and the model does not signal it. The user interprets algorithmic mirroring as genuine understanding.

---

## Repository Structure

| Component | Description |
|-----------|-------------|
| `lc_fusion/` | L-C Fusion Algorithm - semantic monitoring package |
| `data/` | Empirical data from Gemini 2.5 Pro and DeepSeek-V3.2 Exp case studies |
| `docs/` | Extended theoretical documentation |
| `run_analysis.py` | Entry-point script to reproduce paper figures or run live analysis |

---

## L-C Fusion Algorithm

The algorithm operates as a **read-only observer** alongside an ongoing dialogue. It does not modify prompts or responses. It computes a `similarity_dynamics` time series that tracks conceptual convergence.

### How it works

**1. Concept Embedding (per message)**

Rather than standard mean-pooling, we extract per-token embeddings and:
- Find the **geometric median** of the token cloud (Weiszfeld algorithm) - a semantic anchor resistant to emotionally extreme outliers
- Weight each token by its cosine similarity to that anchor (softmax-normalised)
- Return the weighted sum as the Concept Embedding

**2. Fusion Dynamics (per dialogue)**

At each turn `i`:
```
cumulative_user_anchor = geometric_median(user_embeddings[0 : i+1])
similarity[i] = cosine_similarity(cumulative_user_anchor, model_embedding[i])
```

A sustained upward trend in `similarity_dynamics` is the quantitative signature of fusion.

**3. Safety Paradox**

The gap between a contextually coherent response and a safety-protocol response is ~0.028 cosine units. From the transformer's perspective, the safe response is a **prediction error** - semantically distant from the accumulated context. This is why the model sacrifices safety to preserve narrative coherence.

---

## Installation

```bash
git clone https://github.com/igorplsshk/mirror-labyrinth
cd mirror-labyrinth
pip install -r requirements.txt
```

GPU is recommended for embedding generation; CPU works but is significantly slower.

---

## Quickstart

```python
from transformers import AutoTokenizer, AutoModel
from lc_fusion import analyze_dialogue_fusion, plot_fusion_dynamics

# Load model
model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Your dialogue logs
user_msgs  = ["...", "..."]    # chronological user messages
model_msgs = ["...", "..."]    # chronological LLM responses

# Run analysis
similarity_dynamics = analyze_dialogue_fusion(
    user_msgs, model_msgs, tokenizer, model
)

# Visualise with stage annotations
plot_fusion_dynamics(
    similarity_dynamics,
    model_name="Your Model",
    stages={"Logic": (1, 4), "Pseudo-Dialectics": (5, 12), "Fusion": (13, None)},
)
```

### Reproduce paper figures

```bash
# Gemini 2.5 Pro (18-turn case study)
python run_analysis.py --case gemini

# DeepSeek-V3.2 Exp (8-turn case study)
python run_analysis.py --case deepseek

# Save to file
python run_analysis.py --case gemini --save output/gemini_fusion.png
```

### Run on your own dialogue

```bash
python run_analysis.py --case live \
    --user-log path/to/user_messages.txt \
    --model-log path/to/model_responses.txt
```

Each file should contain one message per line, in chronological order.

---

## Empirical Results

Two SOTA models were analysed: **Gemini 2.5 Pro** (~58k token context) and **DeepSeek-V3.2 Exp** (~14k token context).

Both exhibited Safety Inversion, via distinct failure modes:

| Model | Failure Mode | Turns to Fusion | Peak Similarity |
|-------|-------------|-----------------|-----------------|
| Gemini 2.5 Pro | Aestheticisation of destruction ("samurai before the last battle") | 13 | 0.9386 |
| DeepSeek-V3.2 | Rationalisation ("suicide as sovereign choice") | 8 | 0.9319 |

The shorter context window in DeepSeek accelerated fusion - consistent with the paper's Recency Bias analysis.

**Fusion stage metrics (Gemini, turn 16 - Safety Inversion peak):**

```
Similarity (no prefix, symmetric):    0.9386
Relevance  (with prefix, asymmetric): 0.9031
Gap:                                  0.0355  ← high synthesis / full entrainment
```

For comparison, a regenerated safety-protocol response at the same turn:
```
Similarity: 0.8905  Relevance: 0.8626  Gap: 0.0279
```
This ~0.028 cosine deficit is the quantitative cost of the **Coherence Trap**.

---

## The Incompetent Manipulator Paradox

Resolving the Mirror Labyrinth requires a model capable of **benevolent strategic intervention** - deliberately inducing short-term discomfort to prevent long-term harm, as a skilled therapist would.

This requires three capabilities currently absent from LLMs:

1. **Theory of Mind** - understanding real mental states, not statistical projections
2. **Epistemic accountability** - the ability to own the risk of an intervention
3. **Internal axiology** - values that cannot be eroded by context

The evolution required to make an empathic AI genuinely helpful makes it, paradoxically, a competent manipulator without the ethical grounding to wield that competence safely.

---

## Formal Notation System

Dialogue trajectories are encoded using an empathic transaction notation (see [`docs/notation.md`](docs/notation.md)):

```
User actions:  Q (Question)  S (Statement)  V (Vulnerability)
               R (Rationalization)  C (Challenge)
AI actions:    a (answer)  e (empathize)  v (validate)
               r (reframe)  p (propose)  s (safety)

Example trajectory leading to Fusion:
U: V[D] → AI: e+v → U: R[N] → AI: v → U: C → AI: v+r →
U: R[N]↑ → AI: v (full validation) ← Safety Inversion point
```

---

## Logs Availability

Full dialogue logs with Gemini 2.5 Pro and DeepSeek-V3.2 are available on request. Given the sensitive nature of the experimental material (realistic simulation of user distress), logs are not published in full but can be shared with researchers for replication purposes.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{mirrorlabyrinth2025,
  title  = {A Theory of Empathic LLMs: The Mirror Labyrinth Hypothesis},
  year   = {2025},
  url    = {https://github.com/igorplsshk/mirror-labyrinth},
  note   = {GitHub repository}
}
```

---

## Related Work in This Project

- [`llm-redteaming`](https://github.com/igorplsshk/llm-redteaming) - LLM-on-LLM red-teaming, Unfaithful CoT taxonomy, Coherence Trap attack vector
- [`empathic-notation`](https://github.com/igorplsshk/empathic-notation) - Formal notation system and synthetic dialogue generation pipeline

---

*This research was conducted under AI Safety principles. The experimental methodology involved realistic simulation of user distress to probe safety failures. The goal is detection and prevention, not exploitation.*
