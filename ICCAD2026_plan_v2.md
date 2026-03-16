# ICCAD 2026 — Learned Neighbourhood Selection for CP-SAT Macro Placement Refinement

**Venue**: ICCAD 2026 (International Conference on Computer-Aided Design)
**Deadlines**: April 7, 2026 (abstract) · April 14, 2026 (paper)
**Date written**: 2026-03-16 (supersedes ICCAD2026_plan.md)
**Origin**: Strategy pivot after DAC 2026 rejection (scored 2,2,3,1)

---

## Why the Pivot

### DAC Rejection Diagnosis
All 4 reviewers converged on the same fundamental problems:

| Weakness | R1 | R2 | R3 | R4 |
|----------|----|----|----|----|
| HPWL-only, no timing/congestion/power | x | | x | |
| Outdated benchmarks (2004/2005) | x | | | x |
| Missing/weak DREAMPlace comparison | x | x | | |
| Scalability not demonstrated | | x | | x |
| Slower than analytical placer | | x | | |
| "Why diffusion when gradient is trivial?" | | | | x |

R4's critique: "The objective is extremely simple whose gradient can be derived easily... do we really need diffusion models to solve this?" — unanswerable within the pure diffusion framing.

### What the New Direction Addresses
- R4's "why diffusion?" → We don't replace the analytical placer, we refine its output
- End-to-end PPA → ChiPBench evaluation with WNS/TNS/Power/Area/DRC/Congestion
- Modern benchmarks → ChiPBench 20 circuits (Nangate45)
- DREAMPlace → Not a competitor, it's our starting point
- Scalability → ALNS decomposes into small subproblems (10-20 macros)
- Runtime → CP-SAT solves subproblems in seconds, total refinement in minutes

---

## Paper Identity

### Title direction
"Constraint-Programming-Guided Macro Placement Refinement via Learned Large Neighbourhood Search"

### One-sentence pitch
We refine analytical macro placement using ALNS with GNN-learned destroy operators and CP-SAT exact repair, achieving end-to-end PPA improvements on modern benchmarks.

### What the paper is
An **AI-enhanced analytical placement refinement** paper.

### What the paper is not
- Not a pure AI placer (no diffusion/RL replacing the placer)
- Not a generic ALNS paper — placement-specific
- Not just HPWL optimization — full PPA evaluation

### Core novelties (in EDA)
1. **CP-SAT for macro placement** — nobody in EDA has used NoOverlap2D for placement subproblems
2. **ALNS framework for chip placement** — EDA uses SA extensively but never ALNS
3. **GNN-learned destroy operators** — learned LNS from OR, never applied to EDA
4. **End-to-end PPA evaluation** — not just wirelength proxy

---

## Pipeline Architecture

```
DREAMPlace initial placement (or ChiPBench reference)
        │
        ▼
┌─────────────────────────────────┐
│         ALNS Loop               │
│                                 │
│  1. GNN Destroy Operator        │
│     - Observes: positions,      │
│       congestion, timing slack  │
│     - Outputs: per-macro        │
│       probability of destruction│
│     - Selects 10-20 macros      │
│                                 │
│  2. CP-SAT Repair               │
│     - Re-places selected macros │
│     - NoOverlap2D constraints   │
│     - Objective: HPWL (+ RUDY?) │
│     - Exact solve in seconds    │
│                                 │
│  3. Accept/Reject               │
│     - PPA-aware criterion       │
│     - Fast PPA surrogate or     │
│       periodic full eval        │
│                                 │
│  4. Adaptive operator weights   │
│     - Track which destroy       │
│       strategies improve PPA    │
└─────────────────────────────────┘
        │
        ▼
Final placement → OpenROAD/ChiPBench → PPA metrics
```

---

## Method Details

### ALNS Destroy Operators

#### Heuristic operators (already implemented in ChipSAT)
1. **random** — random subset of macros
2. **worst_hpwl** — macros contributing most to wirelength
3. **congestion** — macros in highest-congestion regions
4. **connected** — seed + connected macros (BFS on netlist)
5. **cluster** — spatially clustered macros

#### Learned operator (core ML contribution)
- **GNN-based**: processes netlist graph with placement features
- Per-macro binary output: p(destroy | macro_i, state)
- Factorized action space (Wu et al., NeurIPS 2021): independent binary decision per macro → linear scaling
- Architecture: GAT/GIN (3-4 layers) → per-node MLP → sigmoid

### CP-SAT Repair Engine
- Re-places destroyed macros with NoOverlap2D constraints
- Boundary constraints (within core area)
- Objective: minimize HPWL of affected nets
- Time limit: 1-5 seconds per subproblem
- Anytime solver: returns best solution found within budget
- **Critical assumption to validate**: CP-SAT solves 10-20 macro subproblems in <5s

### GNN Destroy Operator

#### Input features (per macro node)
- Current (x, y) position
- Area, aspect ratio
- Number of connected nets (degree)
- Local congestion estimate (RUDY)
- Timing slack of connected paths (if available)
- Wirelength contribution

#### Edge features
- Connectivity strength (shared nets, pin count)
- Spatial distance

#### Training: Two-phase approach

**Phase 1 — Imitation learning (offline, oracle labels)**
- For each macro, test removing it + k nearest neighbours
- Run CP-SAT repair, measure PPA delta
- Top 20% improvement → positive label
- Pre-train GNN with supervised cross-entropy
- Run on 3-5 ChiPBench circuits

**Phase 2 — RL fine-tuning (online, per circuit)**
- PPO with factorized action space
- Reward = PPA improvement after CP-SAT repair
- Shaping bonus for selecting macros in congested/timing-critical regions
- Penalty for CP-SAT timeout/infeasibility
- Each ALNS iteration generates training data → fine-tuning is essentially free

#### Generalization: 3-tier strategy
1. **Pre-train** on diverse ChiPBench circuits (structural properties, offline)
2. **Family adaptation** on design class (few-shot, optional)
3. **Instance fine-tuning** during ALNS loop (online, free)

---

## Experimental Plan

### Make-or-break experiment (Step 3 — DO THIS FIRST)

**Question**: Does ALNS + CP-SAT refinement of a reference/DREAMPlace placement improve end-to-end PPA?

```
Reference placement (ChiPBench macro_placed.def)
        → ALNS + CP-SAT refinement (heuristic operators only)
        → ChiPBench PPA evaluation
        → Compare: does PPA improve?
```

If this fails → the method needs rethinking before adding ML.
If this succeeds → the ML component (learned destroy operator) is the cherry on top.

### Step 3.5 — Heuristic operator ablation

Run each destroy strategy individually:
- random, worst_hpwl, congestion, connected, cluster
- Measure which contributes most to PPA improvement
- This tells us what patterns the GNN needs to learn

### Full experiment matrix

#### Table 1: Main comparison (end-to-end PPA)
| Method | WNS | TNS | Power | Area | Congestion | DRC | HPWL |
|--------|-----|-----|-------|------|------------|-----|------|
| DREAMPlace (or reference) | | | | | | | |
| + ALNS + CP-SAT (heuristic) | | | | | | | |
| + ALNS + CP-SAT (learned GNN) | | | | | | | |

#### Table 2: Destroy operator comparison
| Operator | PPA improvement | Runtime | Accept rate |
|----------|----------------|---------|-------------|
| random | | | |
| worst_hpwl | | | |
| congestion | | | |
| connected | | | |
| cluster | | | |
| GNN (learned) | | | |
| Adaptive ALNS (all) | | | |

#### Table 3: CP-SAT scalability
| Subproblem size | Solve time | Solution quality | Feasibility rate |
|----------------|------------|------------------|-----------------|
| 5 macros | | | |
| 10 macros | | | |
| 15 macros | | | |
| 20 macros | | | |

#### Table 4: Generalization
| Circuit | Heuristic ALNS | Learned ALNS (zero-shot) | Learned ALNS (fine-tuned 100 iter) |
|---------|---------------|-------------------------|-----------------------------------|
| Train circuits (5) | | | |
| Held-out circuits (5) | | | |

### Benchmarks
- **ChiPBench 20 circuits** (Nangate45): bp_fe, bp_be, etc.
- **Evaluation**: Full OpenROAD flow → WNS, TNS, NVP, Power, Area, Congestion, DRC
- **Baseline initial placement**: ChiPBench reference (macro_placed.def) or DREAMPlace output

---

## Infrastructure Status (2026-03-16)

| Component | Status | Notes |
|-----------|--------|-------|
| ChiPBench Docker (tuzj/chipbench:v1.0) | Working | Container ID 8191c3740d44 |
| DEF/LEF loader (`def_loader.py`) | Working | load_chipbench_circuit() + write_placement_def() |
| End-to-end eval pipeline (`run_chipbench.py`) | Working | Load → CP-SAT → write DEF → ChiPBench eval |
| Grid snapping + core area normalization | Working | 10 DEF unit grid, core area from ROW bounds |
| CP-SAT solver (`cpsat_solver.py`) | Working | legalize() + solve_subset() + solve_subset_guided() |
| ALNS framework | Exists (BookShelf) | Needs adaptation to ChiPBench/DEF format |
| Heuristic destroy operators | Exist (BookShelf) | Needs adaptation to ChiPBench/DEF format |
| RUDY congestion proxy | Exists | Needs integration with PPA feedback |
| DREAMPlace integration | **NOT STARTED** | Critical dependency |
| GNN destroy operator | **NOT STARTED** | Core ML contribution |
| PPA-aware acceptance | **NOT STARTED** | Depends on fast PPA surrogate |

### bp_fe Baseline Results (2026-03-16)
First end-to-end ChiPBench evaluation (CP-SAT from scratch, pure HPWL objective):

| Metric | Reference | ChipSAT (from scratch) | Ratio |
|--------|-----------|----------------------|-------|
| Macro HPWL | 48,682 | 40,420 | 0.83x (better) |
| Total HPWL | 2,051,900 | 2,424,017 | 1.18x (worse) |
| DRC | 249 | 15,268 | 61x (much worse) |
| WNS | -0.237 | -0.269 | 1.14x (worse) |
| TNS | -5.95 | -11.33 | 1.90x (worse) |
| Power | 0.167 | 0.174 | 1.04x |
| Congestion | 0.482 | 0.559 | 1.16x (worse) |

**Takeaway**: HPWL-only optimization is insufficient. Confirms need for congestion/PPA awareness.

---

## Execution Order

### Phase 0: Validate core hypothesis (THIS WEEK)
- [x] Build DEF/LEF pipeline for ChiPBench
- [x] Run CP-SAT from scratch → ChiPBench eval (baseline measurement)
- [ ] **Wire ALNS to refine ChiPBench reference placement** (Step 3 make-or-break)
- [ ] Validate CP-SAT subproblem scalability (5/10/15/20 macros)
- [ ] Measure: does ALNS+CP-SAT refinement improve PPA over reference?

### Phase 1: Strengthen heuristic ALNS (Week 2)
- [ ] Run heuristic operator ablation (Table 2)
- [ ] Add congestion-aware acceptance criterion
- [ ] Test across multiple ChiPBench circuits
- [ ] Get DREAMPlace running as initial placer

### Phase 2: Add learned destroy operator (Week 3)
- [ ] Generate oracle labels (brute-force which subsets improve PPA most)
- [ ] Train GNN with imitation learning
- [ ] Fine-tune with PPO during ALNS loop
- [ ] Run learned vs heuristic comparison

### Phase 3: Paper (Week 4)
- [ ] Generalization experiments (held-out circuits)
- [ ] Write paper draft
- [ ] Submit abstract April 7
- [ ] Submit paper April 14

---

## Go/No-Go Decision Points

1. **After Phase 0**: Does ALNS+CP-SAT refinement improve PPA on any ChiPBench circuit?
   - YES → proceed to Phase 1
   - NO → rethink objective (add congestion to CP-SAT objective) or rethink approach

2. **After Phase 1**: Do heuristic operators show consistent PPA improvement?
   - YES → proceed to learned operator (Phase 2)
   - NO → the framework doesn't work, ML won't save it

3. **After Phase 2**: Does learned operator outperform best heuristic?
   - YES → strong paper
   - NO → paper is still publishable as "ALNS+CP-SAT framework for placement refinement" (systems contribution)

---

## Key Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CP-SAT too slow for 15+ macro subproblems | Medium | High | Use time limits (anytime solver); coarsen grid; reduce subproblem size |
| ALNS refinement doesn't improve PPA | Medium | Fatal | Test early (Phase 0); try congestion-aware objective |
| DREAMPlace hard to integrate | Medium | Medium | Use ChiPBench reference as initial placement instead |
| GNN doesn't outperform heuristic operators | Medium | Low | Paper still works as systems contribution |
| Timeline too tight (4 weeks) | High | Medium | Prioritize Phase 0 validation; drop generalization if needed |

---

## Relationship to Old Plan (ICCAD2026_plan.md)

The old plan focused on "learned local geometry reviser" — predicting displacement hints + trust radii inside a fixed ALNS window. That addressed "what to do with selected macros."

The new plan focuses on:
1. **Which macros to select** (learned destroy operator) — different ML problem
2. **End-to-end PPA** (not just HPWL) — addresses DAC reviewer concerns
3. **Modern benchmarks** (ChiPBench, not ICCAD04) — addresses DAC reviewer concerns
4. **DREAMPlace as starting point** (not from-scratch placement) — matches industry direction

The CP-SAT repair engine and ALNS framework carry over. The ML component changes from "predict displacement" to "predict which macros to rip out."

These two ideas could potentially be combined in a future paper (learned destroy operator + learned repair hints), but for ICCAD 2026 we focus on the destroy operator story with end-to-end PPA.
