"""
Single-sided training environment: opponent uses MCTS internally; agent acts only on its turns.

Follows the same wrapper pattern as the greedy env to keep base env unchanged.
"""
from __future__ import annotations

from typing import Optional, Tuple
import random

from .seal_slammers_env import SealSlammersEnv
from .mcts_agent import MCTSAgent, PPOPolicyAdapter


class SealSlammersSingleSidedMCTSEnv(SealSlammersEnv):
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
        fixed_agent_player_id: Optional[int] = None,
        # MCTS params
        mcts_sims: int = 128,
        mcts_cpuct: float = 1.4,
        mcts_max_depth: int = 3,
        mcts_angle_step: int = 6,
        mcts_strength_topk: int = 2,
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
        self.fixed_agent_player_id = fixed_agent_player_id
        self.agent_player_id: Optional[int] = None
        self._mcts = MCTSAgent(
            n_sims=mcts_sims,
            c_puct=mcts_cpuct,
            max_depth=mcts_max_depth,
            angle_step=mcts_angle_step,
            strength_topk=mcts_strength_topk,
        )
        # Optional: if you want to guide with current PPO during training, set by outer code
        self._policy_adapter: PPOPolicyAdapter | None = None

    def set_policy_adapter(self, adapter: PPOPolicyAdapter | None):
        self._policy_adapter = adapter
        self._mcts.set_policy_adapter(adapter)

    def _choose_agent_side(self):
        if self.fixed_agent_player_id in (0, 1):
            self.agent_player_id = int(self.fixed_agent_player_id)
        else:
            self.agent_player_id = random.choice([0, 1])

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._choose_agent_side()
        self._mcts.on_new_episode()
        if (not self.game.game_over) and (self.game.current_player_turn != self.agent_player_id):
            _obs, _info, _ = self._play_opponent_until_agent_turn()
            obs, info = _obs, _info
        return obs, info

    def step(self, action):
        # If it's not agent's turn, auto-play opponent until agent's turn or terminal
        if not self.game.game_over and self.game.current_player_turn != self.agent_player_id:
            _obs, _info, _ = self._play_opponent_until_agent_turn()
            if self.game.game_over:
                reward_from_agent_perspective = 0.0
                if self.game.winner is not None and self.game.winner != self.agent_player_id:
                    reward_from_agent_perspective = -80.0
                return _obs, reward_from_agent_perspective, True, False, _info

        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        last_obs, last_info, saw_trunc = self._play_opponent_until_agent_turn()
        if self.game.game_over:
            if self.game.winner is not None and self.game.winner != self.agent_player_id:
                reward += -80.0
            return last_obs, reward, True, saw_trunc, last_info

        return last_obs, reward, False, saw_trunc, last_info

    def _play_opponent_until_agent_turn(self) -> Tuple[object, dict, bool]:
        any_truncated = False
        latest_obs, latest_info = None, None
        safety_counter = 0
        while (not self.game.game_over) and (self.game.current_player_turn != self.agent_player_id):
            def base_step(a):
                return SealSlammersEnv.step(self, a)
            action = self._mcts.select_action(self, step_fn=base_step)
            latest_obs, _, terminated, truncated, latest_info = super().step(action)
            any_truncated = any_truncated or bool(truncated)
            safety_counter += 1
            if terminated or truncated or safety_counter > 60:
                break
        if latest_obs is None:
            latest_obs = self._get_obs()
            latest_info = self.get_info()
        return latest_obs, latest_info, any_truncated
