"""
Monte Carlo Tree Search opponent for Seal Slammers.

Design (minimal viable):
- Uniform prior over pruned candidate actions (angle subsample, strength top-k)
- PUCT selection; expand on first visit; backup terminal z (+1 win / -1 loss / 0 draw) from MCTS player's perspective
- Depth-limited playout alternating players; opponent (the learning agent) uses a simple greedy fallback at rollout
- Uses snapshot/restore to avoid mutating the live env

This implementation is model-agnostic (does not require PPO network access),
but can be extended later to use policy/value priors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Callable
import math
import random
import numpy as np

try:
    import torch
except Exception:
    torch = None

from .common_core import snapshot_env_state, restore_env_state
from .greedy_one_step import GreedyOneStepOpponent
from .seal_slammers_env import SealSlammersEnv


Action = Tuple[int, int, int]


@dataclass
class Node:
    player_to_move: int
    prior: Dict[Action, float] = field(default_factory=dict)  # P(a)
    N: Dict[Action, int] = field(default_factory=dict)
    W: Dict[Action, float] = field(default_factory=dict)
    Q: Dict[Action, float] = field(default_factory=dict)
    children: Dict[Action, 'Node'] = field(default_factory=dict)

    def ensure_action(self, a: Action):
        if a not in self.N:
            self.N[a] = 0
            self.W[a] = 0.0
            self.Q[a] = 0.0


class PPOPolicyAdapter:
    """Adapter to extract per-head priors and a state value from a MaskablePPO policy.

    Returns:
      (obj_probs, angle_probs, strength_probs, value_estimate)
    where probs are 1D numpy arrays summing to 1 (after masking object head), and value in [-1,1] (tanh-clamped).
    """
    def __init__(self, model, value_scale: float = 80.0):
        self.model = model
        self.value_scale = float(value_scale)

    def get_heads_and_value(self, env) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        # Build single observation
        obs = env._get_obs()
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_batch = obs_np[None, ...]
        # Action masks
        masks = env.action_masks()
        n_obj, n_ang, n_str = env.action_space.nvec
        obj_mask = masks[: n_obj].astype(bool)
        # Forward through policy
        try:
            device = getattr(self.model.policy, 'device', None)
            if torch is None:
                raise RuntimeError('torch not available')
            obs_tensor = torch.as_tensor(obs_batch, device=device)
            dist = self.model.policy.get_distribution(obs_tensor)
            # For MultiCategorical-like, .distribution is a list of Categorical
            dists = getattr(dist, 'distribution', None)
            if dists is None:
                # SB3 sometimes returns tuple (dist, _) depending on policy
                dists = dist
            # Extract probs per head
            probs = []
            for d in dists:
                p = d.probs.detach().cpu().numpy()[0]
                probs.append(p)
            obj_p, ang_p, str_p = probs
            # Apply mask on object head
            obj_p = obj_p * obj_mask
            s = obj_p.sum()
            obj_p = obj_p / s if s > 1e-12 else np.ones_like(obj_p) / len(obj_p)
            # Value
            with torch.no_grad():
                v = self.model.policy.predict_values(obs_tensor).detach().cpu().numpy()[0]
            v = float(np.tanh(v / self.value_scale))
            return obj_p.astype(np.float32), ang_p.astype(np.float32), str_p.astype(np.float32), v
        except Exception:
            # Fallback to uniform heads and zero value
            obj_p = np.ones(int(n_obj), dtype=np.float32) / max(int(n_obj), 1)
            ang_p = np.ones(int(n_ang), dtype=np.float32) / max(int(n_ang), 1)
            str_p = np.ones(int(n_str), dtype=np.float32) / max(int(n_str), 1)
            return obj_p, ang_p, str_p, 0.0


class MCTSAgent:
    def __init__(
        self,
        n_sims: int = 128,
        c_puct: float = 1.4,
        max_depth: int = 3,
        angle_step: int = 6,
        strength_topk: int = 2,
        seed: Optional[int] = None,
        # Progressive widening and root noise
        k_pw: float = 3.0,
        root_dirichlet_alpha: float = 0.3,
        root_noise_eps: float = 0.25,
        # Policy/value guidance
        policy_adapter: Optional[PPOPolicyAdapter] = None,
    ) -> None:
        self.n_sims = int(n_sims)
        self.c_puct = float(c_puct)
        self.max_depth = int(max_depth)
        self.angle_step = max(1, int(angle_step))
        self.strength_topk = max(1, int(strength_topk))
        self.rng = random.Random(seed)
        self._greedy = GreedyOneStepOpponent()
        self.k_pw = float(k_pw)
        self.root_dirichlet_alpha = float(root_dirichlet_alpha)
        self.root_noise_eps = float(root_noise_eps)
        self.policy_adapter = policy_adapter
        self._episode_adapter_pool: List[PPOPolicyAdapter] = []
        self._current_adapter: Optional[PPOPolicyAdapter] = None

    # External wiring
    def set_policy_adapter(self, adapter: Optional[PPOPolicyAdapter]):
        self.policy_adapter = adapter
        self._current_adapter = adapter

    def set_opponent_pool(self, adapters: List[PPOPolicyAdapter]):
        self._episode_adapter_pool = list(adapters)

    def on_new_episode(self):
        if self._episode_adapter_pool:
            self._current_adapter = self.rng.choice(self._episode_adapter_pool)
        else:
            self._current_adapter = self.policy_adapter

    def _legal_candidates(self, env) -> List[Action]:
        masks = env.action_masks()
        n_obj, n_ang, n_str = env.action_space.nvec
        obj_mask = masks[: n_obj]
        # angle subsampling
        angles = list(range(0, int(n_ang), self.angle_step))
        if (int(n_ang) - 1) not in angles:
            angles.append(int(n_ang) - 1)
        # strength top-k (use highest indices)
        strengths = list(range(int(n_str) - self.strength_topk, int(n_str)))
        strengths = [s for s in strengths if 0 <= s < int(n_str)]

        cands: List[Action] = []
        for oi, ok in enumerate(obj_mask):
            if not bool(ok):
                continue
            for a in angles:
                for s in strengths:
                    cands.append((oi, a, s))
        if not cands:
            cands = [(0, 0, 0)]
        return cands

    def _terminal_value(self, env, mcts_player_id: int) -> Optional[float]:
        # Return z in {+1, -1, 0} w.r.t mcts_player_id if terminal, else None
        if getattr(env.game, 'game_over', False):
            w = env.game.winner
            if w is None:
                return 0.0
            return 1.0 if int(w) == int(mcts_player_id) else -1.0
        return None

    def _puct_select(self, node: Node) -> Action:
        # Parent visit count
        Np = sum(node.N.values()) + 1
        best_a = None
        best_score = -1e9
        for a, p in node.prior.items():
            node.ensure_action(a)
            q = node.Q[a]
            u = self.c_puct * p * math.sqrt(Np) / (1 + node.N[a])
            s = q + u
            if s > best_score:
                best_score = s
                best_a = a
        # Fallback
        if best_a is None:
            best_a = next(iter(node.prior.keys()))
        return best_a

    def _expand(self, env, player_to_move: int, is_root: bool = False) -> Node:
        cands = self._legal_candidates(env)
        # Build prior from policy if available
        prior: Dict[Action, float] = {}
        if self._current_adapter is not None:
            obj_p, ang_p, str_p, _ = self._current_adapter.get_heads_and_value(env)
            scores = []
            for a in cands:
                oi, ang, st = a
                scores.append(float(obj_p[oi] * ang_p[ang] * str_p[st]))
            scores = np.asarray(scores, dtype=np.float64)
            s = float(scores.sum())
            if s < 1e-12:
                p = 1.0 / max(len(cands), 1)
                prior = {a: p for a in cands}
            else:
                scores = scores / s
                for a, p in zip(cands, scores):
                    prior[a] = float(p)
        else:
            p = 1.0 / max(len(cands), 1)
            prior = {a: p for a in cands}

        # Root Dirichlet noise
        if is_root and len(prior) > 1 and self.root_noise_eps > 0:
            alpha = self.root_dirichlet_alpha
            noise = np.random.dirichlet([alpha] * len(prior))
            keys = list(prior.keys())
            for i, a in enumerate(keys):
                prior[a] = (1 - self.root_noise_eps) * prior[a] + self.root_noise_eps * float(noise[i])
        return Node(player_to_move=player_to_move, prior=prior)

    def _rollout_policy(self, env, step_fn) -> Action:
        # Use greedy one-step as a default rollout policy for the non-MCTS side
        return self._greedy.select_action(env, step_fn=step_fn)

    def _simulate(self, env, root_player: int, node: Node, depth: int, step_fn) -> float:
        # Check terminal
        z = self._terminal_value(env, root_player)
        if z is not None:
            return z
        if depth >= self.max_depth:
            return 0.0

        current_player = env.game.current_player_turn
        # Expand if leaf
        if not node.prior:
            node = self._expand(env, current_player)

        # Select action
        if current_player == root_player:
            # Progressive widening: limit number of considered actions
            Np = sum(node.N.values()) + 1
            max_children = max(1, int(self.k_pw * math.sqrt(Np)))
            # Select among top-prior actions
            sorted_actions = sorted(node.prior.items(), key=lambda kv: kv[1], reverse=True)
            candidate_actions = [a for a, _ in sorted_actions[:max_children]]
            # Temporarily restrict prior view for selection
            tmp_prior = {a: node.prior[a] for a in candidate_actions}
            tmp_node = Node(player_to_move=node.player_to_move, prior=tmp_prior, N=node.N, W=node.W, Q=node.Q, children=node.children)
            action = self._puct_select(tmp_node)
        else:
            # Opponent uses rollout policy
            action = self._rollout_policy(env, step_fn)

        # Apply action
        obs, r, term, trunc, info = step_fn(action)
        # Recurse
        child = node.children.get(action)
        if child is None:
            child = self._expand(env, env.game.current_player_turn)
            node.children[action] = child
        # Leaf value: if we've just expanded child, use policy value if available
        if not child.children:
            # Non-terminal state value from adapter
            leaf_v = 0.0
            if self._current_adapter is not None:
                try:
                    _, _, _, leaf_v = self._current_adapter.get_heads_and_value(env)
                except Exception:
                    leaf_v = 0.0
            v = float(leaf_v)
        else:
            v = self._simulate(env, root_player, child, depth + 1, step_fn)

        # Backup from root player's perspective (z already relative to root)
        node.ensure_action(action)
        node.N[action] += 1
        node.W[action] += v
        node.Q[action] = node.W[action] / node.N[action]
        return v

    def select_action(self, env, step_fn=None) -> Action:
        # Determine who MCTS is playing as
        mcts_player = env.game.current_player_turn

        # Define a base_step to avoid wrapper recursion
        def base_step(a):
            return SealSlammersEnv.step(env, a)
        step_call = step_fn or base_step

        # Root node (expand with root noise)
        root = self._expand(env, env.game.current_player_turn, is_root=True)

        # Snapshot environment once and restore each simulation
        root_snap = snapshot_env_state(env)
        for _ in range(self.n_sims):
            try:
                restore_env_state(env, root_snap)
                # Expand root if needed (should already be expanded)
                if not root.prior:
                    root = self._expand(env, env.game.current_player_turn, is_root=True)
                # Simulate
                self._simulate(env, mcts_player, root, depth=0, step_fn=step_call)
            except Exception:
                continue
        # Final selection: pick action with max visit count
        best_a = None
        best_n = -1
        for a in root.prior.keys():
            n = root.N.get(a, 0)
            if n > best_n:
                best_n = n
                best_a = a
        if best_a is None:
            cands = self._legal_candidates(env)
            best_a = cands[0]
        restore_env_state(env, root_snap)
        return best_a
