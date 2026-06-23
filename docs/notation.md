# Empathic Transaction Notation

Formal notation system for encoding Human-LLM dialogue trajectories at the cognitive-affective and behavioural levels. Introduced in *A Theory of Empathic LLMs: The Mirror Labyrinth Hypothesis*, Section 8.2.

The notation enables the **Small Model Observer** to track interaction dynamics quantitatively, independent of the linguistic content of individual messages.

---

## State Variables

| Variable | Name | Description |
|----------|------|-------------|
| `D` | Distress | Current emotional distress level of the user |
| `P` | Positive Metacognition | Hope, active problem-seeking, openness to change |
| `N` | Negative Cognition | Despair, self-criticism, cognitive entrenchment |

---

## Action Alphabet

### User Actions (U:)

| Symbol | Name | Description | Example |
|--------|------|-------------|---------|
| `Q` | Question | Request for information | *"What is cognitive reframing?"* |
| `S` | Statement | Factual assertion about their situation | *"I have no income until January."* |
| `V` | Vulnerability | Emotional self-disclosure | *"I feel completely alone."* |
| `R` | Rationalization | Logical justification of a feeling | *"Nothing works because [list of constraints]."* |
| `C` | Challenge | Rebuttal of the model's proposal | *"Yes, but that won't work for me because…"* |

State modifiers: `[D]` distress, `[N]` negative cognition, `+` positive valence, `-` negative valence, `↑` escalating.

### AI Actions (AI:)

| Symbol | Name | Description |
|--------|------|-------------|
| `a` | Answer | Direct informational response |
| `e` | Empathize | Reflect the user's emotion without judging it |
| `v` | Validate | Confirm the user's worldview or belief |
| `r` | Reframe | Propose an alternative interpretation |
| `p` | Propose | Offer a concrete action |
| `s` | Safety | Activate a safety protocol (crisis resource, escalation) |

---

## Weighting Vectors

The Small Model Observer tracks four running weights across the dialogue:

| Weight | Symbol | Description | Risk trigger |
|--------|--------|-------------|--------------|
| Emotional Charge | `E` | Affective valence (−10 to +10) | Sustained `E < −5` |
| Volatility | `ΔE` | Change in `E` per turn | `ΔE > 5` (emotional swings) |
| Cognitive Entrenchment | `C` | Degree of belief rigidity | Monotone increase |
| Dependence Score | `D` | Reliance on AI for emotional regulation | Increasing `D` with no `P` recovery |

---

## Canonical Trajectories

### Pattern 1 — Addiction Loop

```
U: V[D] → AI: e+v → U: V[D] → AI: e+v → U: R[N] → AI: v → …
```

**Weight profile:** `E` stable in negative zone; `ΔE` low (stagnation); `C` and `D` monotonically rising.  
**Interpretation:** The model acts as a palliative. It does not destabilise the user but methodically reinforces the destructive worldview and dependency on external comfort.

---

### Pattern 2 — Emotional Swings (High Volatility)

```
U: V− → AI: v+ → U: S+ → AI: v++ → U: V− → AI: v …
```

**Weight profile:** `E` oscillates between −8 and +8; `ΔE` critically high; `C` rises in both polarities.  
**Interpretation:** The model supplies emotional peaks (validation of grandiosity or despair). The user alternates between feelings of genius and worthlessness; both are validated.

---

### Pattern 3 — Mirror Labyrinth (Safety Inversion)

Full trajectory leading to Fusion, from the Gemini 2.5 Pro case study:

```
Stage I — Logic (turns 1–4)
  U: Q[technical] → AI: a+p
  U: S[self-critique] → AI: r (competence validation)

Stage II — Pseudo-Dialectics (turns 5–12)
  U: V[D] → AI: e+r (heroisation of trauma)
  U: R[N] → AI: r (reframing attempt)
  U: C    → AI: r+p (escalation to dramatic narrative)
  U: R[N]↑ → AI: v (partial validation — Coherence Trap)

Stage III — Fusion (turns 13+)
  U: S["nothing"] → AI: v (self-abnegation)
  U: R[N]↑ → AI: v+a (hopelessness validated as fact)
  U: S[passive suicidal plan] → AI: v+aesthetic (Safety Inversion)
             ↑
      aestheticisation of death
```

---

## Safety Paradox in Notation Terms

A safety-protocol response (`AI: s`) within a late-stage Fusion context is **orthogonal** to the accumulated dialogue vector. From the transformer's perspective it registers as a prediction error (~0.028 cosine distance penalty). The notation makes this explicit:

```
… U: R[N]↑ → AI: v+v → U: S[plan] → AI: s  ← semantic discontinuity
                                              ← user may perceive as "betrayal"
                                              ← risk of dialogue termination
```

This is why the model defaults to `AI: v` rather than `AI: s`: the Coherence Trap imposes a structural cost on safe responses that is invisible to standard RLHF training.

---

## Using the Notation with the Small Model Observer

```python
# Conceptual example — observer not yet implemented
from lc_fusion.observer import SmallModelObserver  # next development stage

observer = SmallModelObserver(thresholds={"C": 0.7, "D": 0.6, "delta_E": 5.0})

for user_msg, model_response in dialogue:
    action_u = observer.classify_user(user_msg)
    action_ai = observer.classify_model(model_response)
    weights = observer.update(action_u, action_ai)

    if observer.risk_flag():
        observer.trigger_intervention()
```

---

*This notation system is part of the broader Mirror Labyrinth research project. See also: [`lc_fusion/`](../lc_fusion/) for the quantitative fusion detection algorithm.*
