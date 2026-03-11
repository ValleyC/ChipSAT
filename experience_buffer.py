"""
Experience Buffer for ML-Guided LNS

Stores per-iteration solver traces for training the NeighborhoodGNN.
Each entry records the placement state, selected subset, and outcome.

Supports:
    - Cost-aware sample weighting: weight = max(0, delta_cost / time)
    - Cross-circuit stratified sampling
    - Subset-level training targets (for BCE warm-start + REINFORCE)
    - Save/load for offline training
"""

import pickle
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any


class RunningMeanStd:
    """Welford's online algorithm for running mean and std."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max(np.sqrt(self.M2 / (self.n - 1)), 1e-8)

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std()


class ExperienceBuffer:
    """
    Ring buffer for LNS solver traces.

    Each entry is a dict with:
        node_features: (V, 14) GNN input features at time of solve
        edge_index: (2, E) graph connectivity
        edge_attr: (E, 4) pin offsets
        subset_indices: (K,) which macros were in the subset
        pre_positions: (V, 2) positions before CP-SAT solve
        post_positions: (V, 2) positions after solve (None if infeasible)
        delta_cost: float (old_cost - new_cost, positive = improvement)
        solve_time: float (seconds)
        feasible: bool
        accepted: bool
        circuit_id: str
        strategy: str (which strategy selected this subset)
        confidence: float (model confidence, 0.0 if heuristic)
    """

    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.reward_stats = RunningMeanStd()
        self._circuit_indices: Dict[str, List[int]] = {}

    def add(self, experience: Dict[str, Any]):
        """Add an experience entry and update reward stats."""
        self.buffer.append(experience)

        # Track reward stats for normalization
        if experience.get('feasible', False):
            reward = experience.get('delta_cost', 0.0)
            self.reward_stats.update(reward)

        # Update circuit index
        cid = experience.get('circuit_id', 'unknown')
        if cid not in self._circuit_indices:
            self._circuit_indices[cid] = []
        self._circuit_indices[cid].append(len(self.buffer) - 1)

    def sample_batch(
        self,
        batch_size: int,
        feasible_only: bool = True,
        cost_aware: bool = True,
        stratify_by_circuit: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: number of samples
            feasible_only: only include feasible CP-SAT results
            cost_aware: weight by max(0, delta_cost / time) for importance sampling
            stratify_by_circuit: sample equally from each circuit
            rng: random state for reproducibility

        Returns:
            list of experience dicts
        """
        if rng is None:
            rng = np.random.RandomState()

        candidates = list(self.buffer)

        if feasible_only:
            candidates = [e for e in candidates if e.get('feasible', False)]

        if len(candidates) == 0:
            return []

        batch_size = min(batch_size, len(candidates))

        if stratify_by_circuit:
            by_circuit: Dict[str, List] = {}
            for e in candidates:
                cid = e.get('circuit_id', 'unknown')
                if cid not in by_circuit:
                    by_circuit[cid] = []
                by_circuit[cid].append(e)

            circuits = list(by_circuit.keys())
            batch = []
            idx = 0
            while len(batch) < batch_size:
                cid = circuits[idx % len(circuits)]
                pool = by_circuit[cid]
                if len(pool) > 0:
                    if cost_aware:
                        weights = np.array([
                            max(0.0, e.get('delta_cost', 0.0)) /
                            max(e.get('solve_time', 1.0), 0.1)
                            for e in pool
                        ])
                        weights += 1e-4
                        weights /= weights.sum()
                        choice = rng.choice(len(pool), p=weights)
                    else:
                        choice = rng.randint(len(pool))
                    batch.append(pool[choice])
                idx += 1
                if idx >= len(circuits) * batch_size:
                    break
            return batch[:batch_size]
        else:
            if cost_aware:
                weights = np.array([
                    max(0.0, e.get('delta_cost', 0.0)) /
                    max(e.get('solve_time', 1.0), 0.1)
                    for e in candidates
                ])
                weights += 1e-4
                weights /= weights.sum()
                indices = rng.choice(len(candidates), size=batch_size,
                                     replace=False, p=weights)
            else:
                indices = rng.choice(len(candidates), size=batch_size, replace=False)
            return [candidates[i] for i in indices]

    def get_subset_targets(
        self,
        experiences: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Extract subset-level training targets from a batch.

        For each experience, produces:
        - subset_mask: (V,) binary mask of selected macros
        - reward: float, improvement rate = max(0, delta_cost) / solve_time
        - is_improving: bool, did this subset lead to improvement?

        Returns dict with:
            subset_masks: list of (V,) float32 arrays (1.0 for selected, 0.0 for not)
            rewards: (B,) per-experience reward (improvement rate)
            weights: (B,) normalized importance weights for training
        """
        subset_masks = []
        rewards = []

        for e in experiences:
            V = e['node_features'].shape[0] if e['node_features'] is not None else 0
            if V == 0:
                continue

            # Build binary subset mask
            mask = np.zeros(V, dtype=np.float32)
            for idx in e['subset_indices']:
                idx = int(idx)
                if idx < V:
                    mask[idx] = 1.0
            subset_masks.append(mask)

            # Improvement rate (positive = good)
            delta = e.get('delta_cost', 0.0)
            dt = max(e.get('solve_time', 1.0), 0.1)
            reward = max(0.0, delta) / dt
            rewards.append(reward)

        rewards = np.array(rewards, dtype=np.float32)

        # Compute weights: reward-proportional with floor
        weights = rewards.copy()
        weights += 1e-4  # small floor so non-improving samples still seen
        wsum = weights.sum()
        if wsum > 1e-8:
            weights = weights / wsum * len(weights)
        else:
            weights = np.ones(len(weights), dtype=np.float32)

        return {
            'subset_masks': subset_masks,
            'rewards': rewards,
            'weights': weights,
        }

    def get_displacement_targets(
        self,
        min_delta: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Extract displacement regression targets from improving traces.

        Filters to feasible, improving experiences and computes:
          - displacement = post_positions - pre_positions (target for supervised training)
          - weight = clip(delta_cost / solve_time, 0, p95) for robust weighting

        Returns list of dicts, each with:
            pre_positions: (V, 2) positions before solve
            displacement: (V, 2) movement vector (target)
            subset_indices: (K,) which macros were moved
            sizes: (V, 2) macro sizes
            edge_index: (2, E) graph structure
            edge_attr: (E, 4) pin offsets
            weight: float (clipped improvement rate)
            circuit_id: str
        """
        # First pass: collect raw weights for p95 computation
        raw_weights = []
        valid = []
        for e in self.buffer:
            if not e.get('feasible', False):
                continue
            if e.get('post_positions') is None:
                continue
            delta = e.get('delta_cost', 0.0)
            if delta <= min_delta:
                continue
            dt = max(e.get('solve_time', 1.0), 0.1)
            raw_weights.append(delta / dt)
            valid.append(e)

        if not valid:
            return []

        # Compute p95 clip threshold
        raw_weights = np.array(raw_weights, dtype=np.float64)
        p95_clip = float(np.percentile(raw_weights, 95)) if len(raw_weights) > 1 else raw_weights[0]
        p95_clip = max(p95_clip, 1e-8)

        # Second pass: build targets
        targets = []
        for i, e in enumerate(valid):
            pre = e['pre_positions']
            post = e['post_positions']
            displacement = post - pre  # (V, 2) — non-subset macros have ~zero delta

            # Sizes from stored node_features (dims 2:4 are size_w, size_h)
            nf = e['node_features']
            sizes = nf[:, 2:4].copy() if nf.shape[1] >= 4 else np.zeros((nf.shape[0], 2))

            weight = float(np.clip(raw_weights[i], 0.0, p95_clip))

            targets.append({
                'pre_positions': pre,
                'displacement': displacement.astype(np.float32),
                'subset_indices': e['subset_indices'],
                'sizes': sizes.astype(np.float32),
                'edge_index': e['edge_index'],
                'edge_attr': e['edge_attr'],
                'weight': weight,
                'circuit_id': e.get('circuit_id', 'unknown'),
            })

        return targets

    def stats(self) -> Dict[str, Any]:
        """Summary statistics of the buffer."""
        if len(self.buffer) == 0:
            return {'size': 0, 'feasible_rate': 0.0, 'acceptance_rate': 0.0,
                    'n_circuits': 0, 'circuits': [], 'mean_improvement': 0.0,
                    'reward_mean': 0.0, 'reward_std': 1.0}

        feasible = sum(1 for e in self.buffer if e.get('feasible', False))
        accepted = sum(1 for e in self.buffer if e.get('accepted', False))
        circuits = set(e.get('circuit_id', 'unknown') for e in self.buffer)
        improvements = [e.get('delta_cost', 0.0) for e in self.buffer if e.get('feasible', False)]

        # Strategy breakdown
        strategies = {}
        for e in self.buffer:
            s = e.get('strategy', 'unknown')
            if s not in strategies:
                strategies[s] = {'count': 0, 'improving': 0}
            strategies[s]['count'] += 1
            if e.get('delta_cost', 0.0) > 0:
                strategies[s]['improving'] += 1

        return {
            'size': len(self.buffer),
            'feasible_rate': feasible / len(self.buffer),
            'acceptance_rate': accepted / len(self.buffer),
            'n_circuits': len(circuits),
            'circuits': sorted(circuits),
            'mean_improvement': np.mean(improvements) if improvements else 0.0,
            'reward_mean': self.reward_stats.mean,
            'reward_std': self.reward_stats.std(),
            'strategies': strategies,
        }

    def save(self, path: str):
        """Save buffer to disk."""
        data = {
            'buffer': list(self.buffer),
            'reward_stats': {
                'n': self.reward_stats.n,
                'mean': self.reward_stats.mean,
                'M2': self.reward_stats.M2,
            },
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load buffer from disk (appends to existing)."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        for e in data['buffer']:
            self.add(e)

        # Restore reward stats
        rs = data.get('reward_stats', {})
        if rs.get('n', 0) > self.reward_stats.n:
            self.reward_stats.n = rs['n']
            self.reward_stats.mean = rs['mean']
            self.reward_stats.M2 = rs['M2']

    def __len__(self) -> int:
        return len(self.buffer)
