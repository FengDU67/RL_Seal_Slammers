"""
Two-sided MCTS self-play environment wrapper.

Both players act via MCTSAgent. Optional z-only reward mode exposes AlphaZero-like terminal rewards
to the training algorithm (PPO), while the internal game scoring still runs normally.

This wrapper composes on top of SealSlammersEnv by intercepting step() to run opponent moves
and optionally override the external reward.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .seal_slammers_env import SealSlammersEnv
from .mcts_agent import MCTSAgent, PPOPolicyAdapter


class SealSlammersMCTSSelfPlayEnv(SealSlammersEnv):
    def __init__(
        self,
        render_mode=None,
        num_objects_per_player=None,
        p0_hp_fixed=None,
        p0_atk_fixed=None,
        p1_hp_fixed=None,
        p1_atk_fixed=None,
        hp_range=None,
        atk_range=None,
        # MCTS params shared by both sides
        mcts_sims: int = 128,
        mcts_cpuct: float = 1.4,
        mcts_max_depth: int = 3,
        mcts_angle_step: int = 6,
        mcts_strength_topk: int = 2,
        # z-only external reward (optional)
        z_only_reward: bool = False,
    ):
        super().__init__(
            render_mode=render_mode,
            num_objects_per_player=(num_objects_per_player if num_objects_per_player is not None else 3),
            p0_hp_fixed=p0_hp_fixed,
            p0_atk_fixed=p0_atk_fixed,
            p1_hp_fixed=p1_hp_fixed,
            p1_atk_fixed=p1_atk_fixed,
            hp_range=(hp_range if hp_range is not None else (35, 50)),
            atk_range=(atk_range if atk_range is not None else (7, 10)),
        )
        self.z_only_reward = bool(z_only_reward)
        # Build two MCTS agents (can share params; optionally different adapters)
        self._mcts = [
            MCTSAgent(n_sims=mcts_sims, c_puct=mcts_cpuct, max_depth=mcts_max_depth,
                      angle_step=mcts_angle_step, strength_topk=mcts_strength_topk),
            MCTSAgent(n_sims=mcts_sims, c_puct=mcts_cpuct, max_depth=mcts_max_depth,
                      angle_step=mcts_angle_step, strength_topk=mcts_strength_topk),
        ]
        self._adapters: list[PPOPolicyAdapter | None] = [None, None]

    def set_policy_adapters(self, adapter_p0: PPOPolicyAdapter | None, adapter_p1: PPOPolicyAdapter | None):
        self._adapters = [adapter_p0, adapter_p1]
        for i in (0,1):
            self._mcts[i].set_policy_adapter(self._adapters[i])

    def set_opponent_pools(self, pool_p0: list[PPOPolicyAdapter], pool_p1: list[PPOPolicyAdapter]):
        self._mcts[0].set_opponent_pool(pool_p0)
        self._mcts[1].set_opponent_pool(pool_p1)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        for i in (0,1):
            self._mcts[i].on_new_episode()
        # If not starting player is P0, we still return obs for the current player as usual
        return obs, info

    def step(self, action):
        # Determine which player is acting (PPO controls current_player_turn)
        player = self.game.current_player_turn
        # Let PPO-provided action pass through MCTS? In self-play, we want both sides MCTS:
        # So ignore external action and instead pick via MCTS for the current player.
        def base_step(a):
            return SealSlammersEnv.step(self, a)
        chosen = self._mcts[player].select_action(self, step_fn=base_step)
        obs, reward, terminated, truncated, info = super().step(chosen)

        # If z-only external reward is enabled and episode ended, replace reward with +1/-1/0 for current player
        if self.z_only_reward:
            if terminated or truncated:
                if self.game.winner is None:
                    reward = 0.0
                else:
                    reward = 1.0 if int(self.game.winner) == int(player) else -1.0
            else:
                reward = 0.0
        return obs, reward, terminated, truncated, info
