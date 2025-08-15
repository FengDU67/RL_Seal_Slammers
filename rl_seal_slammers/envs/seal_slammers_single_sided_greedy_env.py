"""
Single-sided training environment: agent only acts on its turns, opponent uses one-step greedy internally.

Keeps the base environment logic intact by inheriting from SealSlammersEnv and
only adding opponent auto-play between agent turns. This way, training can simply
switch to this Env class to use the greedy opponent.
"""
from __future__ import annotations

from typing import Optional, Tuple
import random

from .seal_slammers_env import SealSlammersEnv
from .greedy_one_step import GreedyOneStepOpponent
from .common_core import evaluate_action_reward


class SealSlammersSingleSidedGreedyEnv(SealSlammersEnv):
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
        fixed_agent_player_id: Optional[int] = None,  # if None, randomize each episode
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
        self._greedy = GreedyOneStepOpponent()

    def _choose_agent_side(self):
        if self.fixed_agent_player_id in (0, 1):
            self.agent_player_id = int(self.fixed_agent_player_id)
        else:
            self.agent_player_id = random.choice([0, 1])

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._choose_agent_side()
        # If it's not agent's turn at the start, let greedy opponent play until it's agent's turn or game ends
        if not self.game.game_over and self.game.current_player_turn != self.agent_player_id:
            _obs, _info, _ = self._play_opponent_until_agent_turn()
            # return observation for agent's upcoming turn (or terminal)
            obs, info = _obs, _info
        return obs, info

    def step(self, action):
        # Agent should only act on its own turns; if not, we bring the state to agent turn.
        if not self.game.game_over and self.game.current_player_turn != self.agent_player_id:
            _obs, _info, _ = self._play_opponent_until_agent_turn()
            # If the game ended during opponent auto-play, return a zero-reward terminal from agent perspective
            if self.game.game_over:
                # Synthesize terminal penalty for agent if lost on opponent's turn
                reward_from_agent_perspective = 0.0
                if self.game.winner is not None and self.game.winner != self.agent_player_id:
                    # Mirror LOSS_PENALTY magnitude from base env
                    reward_from_agent_perspective = -80.0
                terminated = True
                truncated = False
                return _obs, reward_from_agent_perspective, terminated, truncated, _info

        # Now it's agent's turn; execute the agent action normally
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # After agent acts, let greedy opponent play until it's agent's turn again or episode ends
        last_obs, last_info, saw_trunc = self._play_opponent_until_agent_turn()
        if self.game.game_over:
            # If opponent won during its internal turn, add terminal loss signal for agent (the base env gives win to opponent only)
            if self.game.winner is not None and self.game.winner != self.agent_player_id:
                reward += -80.0  # mirror LOSS_PENALTY
            return last_obs, reward, True, saw_trunc, last_info

        # Otherwise, return the next agent-turn observation; reward is only from the agent action
        return last_obs, reward, False, saw_trunc, last_info

    def _play_opponent_until_agent_turn(self) -> Tuple[object, dict, bool]:
        """Auto-play opponent with greedy policy until it's agent's turn or episode ends.
        Returns (observation, info, any_truncated_during_auto_play).
        """
        any_truncated = False
        latest_obs, latest_info = None, None
        safety_counter = 0
        # Continue while not agent turn and game not finished
        while (not self.game.game_over) and (self.game.current_player_turn != self.agent_player_id):
            # If opponent cannot move, base env will handle round switching or termination via step()
            # Select action using greedy with a step_fn that calls the base env.step
            def base_step(a):
                return SealSlammersEnv.step(self, a)
            action = self._greedy.select_action(self, step_fn=base_step)
            latest_obs, _, terminated, truncated, latest_info = super().step(action)
            any_truncated = any_truncated or bool(truncated)
            safety_counter += 1
            if terminated or truncated or safety_counter > 50:
                break
        # Fallback to current observation if we didn't step (e.g., already agent turn)
        if latest_obs is None:
            latest_obs = self._get_obs()
            latest_info = self.get_info()
        return latest_obs, latest_info, any_truncated
