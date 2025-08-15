"""
One-step greedy opponent policy for Seal Slammers.

Strategy:
- Always use maximum strength (strength_idx = max index)
- Try all 72 angle bins for each legal movable object of the current player
- Simulate an env.step() per candidate, collect the immediate reward, pick the best
- Uses common_core.snapshot/restore to avoid mutating the live env

Usage:
    from rl_seal_slammers.envs.greedy_one_step import GreedyOneStepOpponent
    bot = GreedyOneStepOpponent()
    action = bot.select_action(env)

This keeps the base game/env unchanged and can be reused by different wrappers.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .common_core import snapshot_env_state, restore_env_state, evaluate_action_reward


class GreedyOneStepOpponent:
    def __init__(self, angle_bins: int = 72, strength_levels: int = 5):
        # Defaults are only used if env.action_space doesn't provide nvec (shouldn't happen)
        self.angle_bins = angle_bins
        self.strength_levels = strength_levels

    def select_action(self, env, step_fn=None) -> Tuple[int, int, int]:
        """Return a legal action (object_idx, angle_idx, strength_idx) maximizing immediate reward.

        - If no legal actions exist (shouldn't happen normally), returns a safe default (0, 0, 0)
        - Respects env.action_masks() for object availability
        - Max strength by design; we sweep angle_idx in [0, angle_bins)
        """
        # Prepare legality masks
        masks = env.action_masks()
        # Pull dimensions from env to avoid mismatch
        n_obj, n_ang, n_str = env.action_space.nvec
        object_mask = masks[:n_obj]
        # We don't need angle/strength masks individually because for our env they are all True

        max_strength_idx = int(n_str) - 1

        best_action = (0, 0, 0)
        best_reward = -np.inf

        # Iterate legal objects
        for obj_idx, ok in enumerate(object_mask):
            if not bool(ok):
                continue
            # Sweep all angles at max strength
            for angle_idx in range(int(n_ang)):
                candidate = (obj_idx, angle_idx, max_strength_idx)
                r = evaluate_action_reward(env, candidate, step_fn=step_fn)
                if r > best_reward:
                    best_reward = r
                    best_action = candidate

        return best_action
