# ICCAD 2026 — Learned Local Reviser for CP-SAT Exact Placement Repair

**Venue**: ICCAD 2026 (International Conference on Computer-Aided Design)
**Deadlines**: April 7, 2026 (abstract) · April 14, 2026 (paper)
**Date written**: 2026-03-11

---

## Paper Identity

### One-sentence pitch
We introduce a learned local geometric prior for exact placement repair: given a fixed ALNS-selected local window, a lightweight model predicts bounded geometry-aware corrections and trust regions that make CP-SAT local repair faster and more effective under equal wall-clock budget.

### What the paper is
A **placement-specific local refinement** paper.

### What the paper is not
- Not a generic RL-for-search paper
- Not a generic OR-Tools acceleration paper
- Not a full-chip neural placer
- Not a neighborhood-selection paper

This positioning ensures the contribution is judged as a **placement method**, not as a generic search-control method.

### Candidate titles
- *Learning Local Geometry Priors for Exact Chip Placement Repair*
- *Solver-Aware Local Revision for CP-SAT-Based Placement Refinement*
- *Trust-Region Local Revision for Exact Placement Repair*

---

## Novelty Claim (narrow and defensible)

> This work introduces a placement-specific normalized local reviser that predicts bounded geometric corrections and trust regions for exact CP-SAT local repair inside a fixed ALNS framework, improving repair efficiency and final placement quality under equal wall-clock time.

**Key phrase for the paper**: *learning compresses the exact local search space.*
This is stronger than "learning provides hints."

---

## Four Testable Claims

These are the only claims the paper should make, and all four must be experimentally supported.

| # | Claim |
|---|-------|
| 1 | A normalized local geometric model transfers across unseen circuits better than chip-specific coordinate learning. |
| 2 | Learned trust-region guidance reduces CP-SAT effort more than hints alone. |
| 3 | The learned prior beats simple heuristic priors under equal wall-clock. |
| 4 | The gains persist when the outer ALNS policy and windows are held fixed. |

---

## What NOT to Claim
- First use of ML with ALNS
- First use of CP-SAT for placement
- First learning-guided local search
- General combinatorial-optimization breakthrough

---

## Reviewer Concerns and Answers

| Concern | Defense |
|---------|---------|
| "CP-SAT does the real work" | Freeze outer ALNS; compare CP-SAT alone vs heuristic prior vs learned prior on the **same windows** |
| "Just a warm-start trick" | Separate hints-only vs trust-regions-only vs hints+trust-regions |
| "Simple heuristics could do this" | **Simplicity challenge table**: centroid pull, boundary-avoidance, local force relaxation, net-weighted anchor pull, hand-crafted trust box |
| "Neighborhood control in disguise" | Hold outer ALNS windows fixed in core ablations; model acts only **inside** chosen window |
| "Off-topic from placement" | All representation, loss, metrics are placement-native: geometry, anchors, sizes, net connectivity, legality, HPWL/congestion/runtime |
| "Doesn't generalize" | Normalize local subproblems; train on windows not whole chips; evaluate on held-out circuits |

---

## Method

### Outer Loop (fixed, unchanged from Phase 1)

ALNS does:
- Subset/window selection
- Outer search schedule
- Acceptance/rejection by true cost

The learned model does **not** choose neighborhoods in the first paper.

### Inner Learned Component

Inside a chosen local window, the model predicts:
- `delta_mean` — per-movable-node displacement proposal (required)
- `trust_radius` — per-node or per-window trust radius (required)
- `predicted_gain` — optional, v2 only
- `predicted_runtime` — optional, v2 only
- `predicted_feasibility` — optional, v2 only

> **Do not add optional heads for ICCAD v1.** Each extra head requires re-collected labeled data and separate validation. If `delta_mean + trust_radius` doesn't win, extra heads won't save it.

### CP-SAT Integration (4 modes)

| Mode | Description | Expected value |
|------|-------------|----------------|
| Hints only | Revised positions as solver hints | Modest |
| Trust regions only | Constrain domain around predicted position | Strong if done well |
| Hints + trust regions | **Primary method** | Best |
| Confidence-based freezing | Freeze low-motion high-confidence nodes | Real solve-time reduction if confidence is reliable |

**The key mechanism**: learned output reduces search entropy, not merely suggests a solution.

---

## Local Subproblem Formulation

### Extraction
For each ALNS-selected subset:
1. Compute current subset bounding box
2. Expand by `window_fraction`
3. Include movable subset nodes
4. Include in-window anchors (simple distance-based for v1)
5. Include capped number of out-of-window directly connected anchors

**Node budget**: 64 nodes cap for v1. Do not implement composite anchor scoring for ICCAD — simple distance is sufficient.

### Normalization (enables cross-circuit transfer)
- Center window at origin
- Scale by max span
- Canonicalize long axis
- Augment by flips and 180° rotation
- Store inverse transform for decoding

### Node Features (6D — keep this for v1, do not expand)
| Feature | Description |
|---------|-------------|
| x, y | Normalized position |
| w, h | Macro size in local scale |
| is_movable | 0 (anchor) or 1 (subset) |
| degree | Incident nets, normalized |

### Edge Features (4D)
- pin_dx, pin_dy per endpoint in local coordinates

### Displacement Target (per movable node)
```
delta_local = (post_pos[i] - pre_pos[i]) / local_scale
```

### Trust Radius Target (per movable node)
```
trust_target = clip(1.2 * max(|delta_local_x|, |delta_local_y|) / window_fraction_local, 0.05, 1.0)
```

---

## Model

### Architecture
- Local message-passing GNN (NetSpatialGNN reused, 6D input)
- `trust_radius_head`: `ReluMLP([hidden, hidden//2]) → Linear(1) → Sigmoid`
- Modest hidden size (~64-128), low inference cost
- Architecture is **not** the main contribution — keep it simple

### Loss
```python
disp_loss  = MSE(disp_pred[movable], disp_target[movable])
trust_loss = MSE(trust_pred[movable], trust_target[movable])
loss = weight * (disp_loss + 0.5 * trust_loss)
```

### Training Stage Summary

| Stage | Method | Status |
|-------|--------|--------|
| 0 | Tune and freeze teacher (ALNS + CP-SAT) | Complete (Phase 1 sweep done) |
| 1 | Collect local traces (3 circuits × 3 seeds) | Complete |
| 2 | Supervised pretraining (displacement + trust) | In progress (training epoch 1/100) |
| 3 | Optional RL fine-tuning | Only if supervised wins |

---

## Baselines (Priority Order)

1. **Placement-specific local refinement / exact repair baselines** (primary)
2. **Learned placement-improvement baselines**
3. **Heuristic local priors** (must include for Concern C defense)
4. **Learning-guided ALNS / operator-selection** (short contrast paragraph only)

Do not allocate major paper space to generic RL-for-local-search literature.

---

## Experiment Matrix

### 10.1 Main Comparison Table (equal wall-clock)

| Method | HPWL | Congestion | Runtime | Success rate |
|--------|------|------------|---------|--------------|
| Tuned ALNS + CP-SAT (teacher) | | | | |
| Teacher + heuristic prior | | | | |
| Teacher + learned prior (ours) | | | | |

### 10.2 Mechanism Table (fixed windows, fixed outer loop)

| Condition | Hints | Domain center | Trust radius | Description |
|-----------|-------|---------------|--------------|-------------|
| `pure` | None | Current pos | Uniform | Baseline |
| `random_hints` | Random ±0.5w | Random hint | Uniform | Do learned hints matter? |
| `hint_only` | GNN | GNN hint | Uniform | Value of displacement |
| `trust_only` | None | Current pos | GNN | Value of trust restriction |
| `hint_plus_trust` | GNN | GNN hint | GNN | Full system |

### 10.3 Fixed-Window Ablation
Hold chosen local windows fixed across all methods. Directly enforces the "inside the window" intervention-point claim.

### 10.4 Simplicity Challenge Table (most important for reviewer defense)

| Prior | HPWL | Branches/call | Description |
|-------|------|---------------|-------------|
| Centroid pull | | | Move toward local centroid |
| Boundary-avoidance | | | Push toward interior |
| Local force relaxation | | | Spring forces from nets |
| Net-weighted anchor pull | | | Pull toward high-degree anchors |
| Hand-crafted trust box | | | Fixed small box around current pos |
| Learned prior (ours) | | | GNN displacement + trust |

**This table is the make-or-break experiment.** Run it before writing any other results section.

### 10.5 Generalization Table (held-out circuits)

Train on: ibm01, ibm03, ibm07
Test on: ibm12, ibm15 (and optionally ibm02, ibm04)

| Circuit | Pure CP-SAT | Learned prior | Ratio | Transfer? |
|---------|-------------|---------------|-------|-----------|
| ibm12 | | | | |
| ibm15 | | | | |

---

## Metrics (do not rely on HPWL alone)

- Final placement HPWL
- Congestion / overflow proxy (RUDY)
- Total wall-clock time
- CP-SAT time share
- Average local-solve time
- Feasible local-solve rate
- Accepted local-move rate
- **Branches/call** (from solver telemetry — proves search compression)
- **Conflicts/call** (from solver telemetry)
- Gain per CP-SAT second
- Improvement rate (impr/s)

---

## Related Work Structure (ICCAD-native ordering)

1. **Placement-specific local refinement / legalization / exact repair** — primary section
2. **Learned methods for placement improvement** — especially refinement methods
3. **Learning-guided destroy/repair or ALNS control** — short contrast paragraph
4. **Generic RL-for-neighborhood-selection** — very short defensive paragraph

---

## Paper Outline

1. **Introduction** — Motivate exact local repair as powerful but expensive; state local-prior idea and contributions
2. **Background and related work** — Placement-first, short adjacent contrast to learned search-control
3. **Problem setup** — Define local repair inside fixed ALNS windows and exact-repair objective
4. **Method** — Subproblem extraction, normalization, model, supervised training, CP-SAT integration
5. **Experimental setup** — Benchmarks, teacher tuning, equal-wall-clock protocol, metrics
6. **Results** — Main comparison → mechanism table → simplicity challenge → generalization → ablations
7. **Conclusion** — Very short

---

## Implementation Status

| Component | File | Status |
|-----------|------|--------|
| CP-SAT hint-centered domains | `ChipSAT/cpsat_solver.py` — `solve_subset_guided()` | Complete |
| Trust radius head | `ChipSAT/net_spatial_gnn.py` | Complete |
| Collect/train/eval pipeline | `ChipSAT/train_local_reviser.py` | Complete |
| Phase 1 sweep (teacher tuning) | `SDDS-PyTorch/sweep_alns.py` | Complete (666/666 runs) |
| Best teacher config | `ss=10, wf=0.1, tl=0.3` (mean ratio 1.100×) | Identified |
| Training data collected | `local_reviser_data/train.pt` (6770 instances) | Complete |
| Supervised training | 100 epochs on 3 circuits × 3 seeds | **In progress** |
| Heuristic prior baselines | `train_local_reviser.py` — not yet implemented | **TODO** |
| 5-condition ablation | `--mode eval` | Pending training |
| Held-out circuit eval | ibm12, ibm15 | Pending training |

---

## Execution Order (4-week sprint)

### Week 1 (March 11–18)
- [ ] Let training finish (100 epochs)
- [ ] Run 5-condition ablation on ibm01, ibm03, ibm07
- [ ] Implement heuristic prior baselines (centroid pull, force relaxation, hand-crafted trust box)

### Week 2 (March 18–25)
- [ ] Run simplicity challenge table — this determines whether the paper is strong
- [ ] Collect traces on ibm12, ibm15 for generalization eval
- [ ] Train generalization model (same 100 epochs)

### Week 3 (March 25 – April 1)
- [ ] Run held-out circuit evaluation
- [ ] Run fixed-window ablation
- [ ] Determine which claims are supported by data
- [ ] Draft paper around supported claims only

### Week 4 (April 1–7)
- [ ] Finalize results tables
- [ ] Write full paper draft
- [ ] Submit abstract by April 7

### Final stretch (April 7–14)
- [ ] Polish paper
- [ ] Do NOT add RL unless supervised clearly wins and fine-tuning clearly helps
- [ ] Submit by April 14

---

## Go/No-Go Standard

The paper is ICCAD-worthy if all four hold:

1. Learned prior is **placement-specific** (local normalization enables transfer)
2. It improves **exact local repair efficiency** (branches/conflicts drop)
3. Gain survives against **simple heuristic priors** (simplicity challenge table)
4. Effect holds when **outer ALNS windows are fixed** (fixed-window ablation)

If the simplicity challenge table shows a heuristic prior matches the GNN, the paper needs a different angle — either stronger gains elsewhere, or a circuit-scale count showing breadth of improvement.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Trust metric doesn't converge (trust=0.189 at step 1354) | Medium | Check convergence at epoch 20; reduce trust loss weight if needed |
| Simple heuristic prior matches GNN | Medium | Try more training data (more circuits); or pivot to showing solver telemetry (branches) as primary metric |
| Generalization fails (ibm12, ibm15) | Medium | Local normalization is the key; if it fails, investigate normalization quality |
| Timeline too tight | High | Prioritize simplicity challenge table and 5-condition ablation; drop generalization experiments if needed |
| "exact repair" framing challenged | Low | Use "constraint-guaranteed local repair" or "certified-feasible local repair" instead |
