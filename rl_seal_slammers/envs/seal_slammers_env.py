import os  # Added
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random  # Ensure random is imported
from rl_seal_slammers.physics_utils import simulate_projected_path  # Shared trajectory prediction
from .game_core import (
    Game,
    GameObject,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FPS,
    BLACK,
    WHITE,
    RED,
    BLUE,
    GREEN,
    GREY,
    LIGHT_GREY,
    OBJECT_RADIUS,
    OBJECT_DEFAULT_HP,
    OBJECT_DEFAULT_ATTACK,
    NUM_OBJECTS_PER_PLAYER,
    INITIAL_HP,
    INITIAL_ATTACK,
    DEFAULT_HP_RANGE,
    DEFAULT_ATK_RANGE,
    MAX_PULL_RADIUS_MULTIPLIER,
    FORCE_MULTIPLIER,
    MAX_LAUNCH_STRENGTH_RL,
    FRICTION,
    MIN_SPEED_THRESHOLD,
    ELASTICITY,
    MAX_VICTORY_POINTS,
    DAMAGE_COOLDOWN_FRAMES,
    HP_BAR_WIDTH,
    HP_BAR_HEIGHT,
)

# --- Reward scaling constants (normalized magnitudes) ---
KO_POINT_REWARD = 30.0           # per score point gained
WIN_REWARD = 80.0
LOSS_PENALTY = -80.0
INVALID_ACTION_PENALTY = -15.0
SELECTION_SHAPING_REWARD = 0.5   # scaled by shaping_scale
POSITION_SHAPING_REWARD = 0.2    # scaled by shaping_scale
# Adjusted: encourage contact slightly more
COLLISION_SHAPING_REWARD = 2.0   # scaled by shaping_scale (no damage)
# Adjusted distance shaping cap
DIST_REDUCTION_MAX_SHAPING = 3.0 # cap for distance shaping per action (scaled)
ALIGNMENT_MAX_SHAPING = 0.5      # scaled by shaping_scale
DAMAGE_BASE_PER_HP = 1.0         # core signal (unscaled)
# Tier multipliers relative to attacker_atk thresholds (unscaled core bonuses)
DAMAGE_TIER_MULTIPLIERS = {
    1: 1.2,   # >= 1 * atk
    2: 2.0,   # >= 2 * atk
    3: 3.0    # >= 3 * atk
}
TIME_STEP_PENALTY = -0.05        # lighter time penalty
DAMAGE_TAKEN_COEFF = 0.5  # Semi-zero-sum coefficient (alpha) for damage_taken penalty
# New constants for improved meaningless action handling
MEANINGLESS_DAMAGE_THRESHOLD = 0.5  # effective minimal inflicted to count as meaningful
MEANINGLESS_GRACE_STEPS = 2         # first K consecutive meaningless steps are free
MEANINGLESS_BASE = 2.0              # starting penalty after grace
MEANINGLESS_STEP = 0.5              # incremental increase per extra meaningless action
MEANINGLESS_MAX = 6.0               # cap

# --- SealSlammersEnv Class (Gym Environment) ---
class SealSlammersEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}
    # Track active human-display environments to safely decide when to quit pygame fully
    _active_display_envs = 0

    def __init__(self, render_mode=None, num_objects_per_player=NUM_OBJECTS_PER_PLAYER,
                 p0_hp_fixed=None, p0_atk_fixed=None,
                 p1_hp_fixed=None, p1_atk_fixed=None,
                 hp_range=DEFAULT_HP_RANGE,
                 atk_range=DEFAULT_ATK_RANGE):
        super().__init__()
        self.render_mode = render_mode
        
        self.game = Game(
            headless=(render_mode != "human" and render_mode != "rgb_array"),
            num_objects_per_player=num_objects_per_player
            # HP/ATK are not passed to Game constructor anymore
        )
        
        self.window_surface = None
        self.clock = None
        
        # If we started in human mode (Game opened a display), count this env
        if self.render_mode == 'human':
            SealSlammersEnv._active_display_envs += 1
        
        self.num_objects_per_player = num_objects_per_player

        # Store fixed HP/ATK values and ranges
        self.p0_hp_fixed = p0_hp_fixed
        self.p0_atk_fixed = p0_atk_fixed
        self.p1_hp_fixed = p1_hp_fixed
        self.p1_atk_fixed = p1_atk_fixed
        self.hp_range = hp_range
        self.atk_range = atk_range
        
        self.last_opponent_action = [-1.0, -1.0, -1.0] 
        self.consecutive_meaningless_actions = 0
        # NEW: shaping scale (1.0 full shaping -> decays later via callback)
        self.shaping_scale = 1.0

        # Action space: 统一为 72 个方向、5 档力度
        # action[0]: object_to_move (0 to num_objects_per_player - 1)
        # action[1]: angle_idx (0 to 71 for 72 directions)
        # action[2]: strength_idx (0 to 4 for 5 levels)
        self.action_space = spaces.MultiDiscrete([
            num_objects_per_player,
            72,                     # 72 directions
            5                       # 5 strength levels
        ])

        # Observation space:
        # For each object (num_objects_per_player * 2 players):
        #   - Raw x position (obj.x)
        #   - Raw y position (obj.y)
        #   - Raw HP (obj.hp)
        #   - Raw Attack (obj.attack)
        #   - Has moved this turn (1.0 if obj.has_moved_this_turn else 0.0)
        # (5 features per object)
        # Global features:
        #   - Current player turn (0 or 1) (1 feature)
        #   - Player 0 score (raw) (game.scores[0]) (1 feature)
        #   - Player 1 score (raw) (game.scores[1]) (1 feature)
        # Last opponent action features:
        #   - Opponent's chosen object index (1 feature)
        #   - Opponent's chosen angle index (1 feature)
        #   - Opponent's chosen strength index (1 feature)
        # Total features = (num_objects_per_player * 2 * 5) + 1 + 2 + 3
        obs_dim = (num_objects_per_player * 2 * 5) + 1 + 2 + 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # REMOVED return statement from __init__

    def _get_obs(self):
        # 改进的状态表示：使用相对位置和归一化特征
        state = []
        
        # 获取当前玩家的对象作为参考点
        current_player_objects = self.game.players_objects[self.game.current_player_turn]
        opponent_objects = self.game.players_objects[1 - self.game.current_player_turn]
        
        # 计算当前玩家对象的中心位置作为参考点
        current_alive_objects = [obj for obj in current_player_objects if obj.hp > 0]
        if current_alive_objects:
            center_x = sum(obj.x for obj in current_alive_objects) / len(current_alive_objects)
            center_y = sum(obj.y for obj in current_alive_objects) / len(current_alive_objects)
        else:
            center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        
        # 为每个对象编码特征
        for player_objs in self.game.players_objects:
            for obj in player_objs:
                # 相对位置 (归一化到 [-1, 1])
                rel_x = (obj.x - center_x) / (SCREEN_WIDTH / 2)
                rel_y = (obj.y - center_y) / (SCREEN_HEIGHT / 2)
                
                # 归一化HP和攻击力
                norm_hp = obj.hp / 100.0  # 假设最大HP约为100
                norm_attack = obj.attack / 20.0  # 假设最大攻击力约为20
                
                # 移动状态
                has_moved = 1.0 if obj.has_moved_this_turn else 0.0
                
                state.extend([rel_x, rel_y, norm_hp, norm_attack, has_moved])

        # 当前玩家回合
        state.append(float(self.game.current_player_turn))

        # 归一化分数
        state.extend([self.game.scores[0] / 10.0, self.game.scores[1] / 10.0])

        # 上一个对手动作 (归一化)
        norm_last_action = [
            self.last_opponent_action[0] / self.num_objects_per_player,  # 对象索引
            self.last_opponent_action[1] / 72.0,  # 角度索引 (72)
            self.last_opponent_action[2] / 5.0    # 力度索引 (5)
        ]
        state.extend(norm_last_action)

        return np.array(state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # --- (Re)Initialization Logic ---
        # This logic runs on both initial creation and reset.
        # If you need different behavior on first init vs reset, add flags or check self.num_steps.

        # Determine HP and Attack for players
        p0_hp = self.p0_hp_fixed if self.p0_hp_fixed is not None else random.uniform(self.hp_range[0], self.hp_range[1])
        p0_atk = self.p0_atk_fixed if self.p0_atk_fixed is not None else random.uniform(self.atk_range[0], self.atk_range[1])
        p1_hp = self.p1_hp_fixed if self.p1_hp_fixed is not None else random.uniform(self.hp_range[0], self.hp_range[1])
        p1_atk = self.p1_atk_fixed if self.p1_atk_fixed is not None else random.uniform(self.atk_range[0], self.atk_range[1])
        
        # Reinitialize game state with potentially randomized or fixed stats
        self.game.reset_game_state(p0_hp=p0_hp, p0_atk=p0_atk, 
                                  p1_hp=p1_hp, p1_atk=p1_atk)
        
        self.consecutive_meaningless_actions = 0 # Reset counter
        self.last_opponent_action = [-1.0, -1.0, -1.0] # Reset opponent action
        # New: per-player HP snapshots for delayed damage_taken (B2)
        self._hp_snapshot_since_last_turn = {0: [obj.hp for obj in self.game.players_objects[0]],
                                             1: [obj.hp for obj in self.game.players_objects[1]]}

        # --- Observation Construction ---
        observation = self._get_obs()
        # 在 reset 方法中调用 get_info() 以确保初始状态也包含动作掩码
        info = self.get_info() 
        
        # Initialize episode component tracker
        self._episode_components = {}

        if self.render_mode == "human":
            self.render()
        return observation, info

    def action_masks(self):
        """
        Required method for MaskablePPO. Returns action masks for the current state.
        For MultiDiscrete action spaces, this needs to return concatenated masks for each sub-action.
        """
        try:
            current_player_id = self.game.current_player_turn # 当前轮到行动的玩家

            # 1. 创建对象选择掩码 (object_mask)
            object_mask_list = [False] * self.num_objects_per_player
            
            # 只有当当前玩家有棋子可以移动时，才计算有效的对象掩码
            if self.game.can_player_move(current_player_id):
                player_objects = self.game.players_objects[current_player_id]
                for i, obj in enumerate(player_objects):
                    # 对象可选的条件是：存活(hp > 0) 且 本回合尚未移动
                    if obj.hp > 0 and not self.game.object_has_moved_this_turn(current_player_id, i):
                        object_mask_list[i] = True
            
            object_mask = np.array(object_mask_list, dtype=bool)

            # 2. 创建角度掩码 - 如果有有效对象，则所有角度都可选
            angle_mask = np.full(self.action_space.nvec[1], any(object_mask_list), dtype=bool)

            # 3. 创建力度掩码 - 如果有有效对象，则所有力度都可选  
            strength_mask = np.full(self.action_space.nvec[2], any(object_mask_list), dtype=bool)
            
            # 对于MultiDiscrete，掩码应该是各个子动作空间掩码的连接
            # 总长度 = sum(nvec) = 3 + 72 + 5 = 80
            concatenated_mask = np.concatenate([object_mask, angle_mask, strength_mask])
            
            return concatenated_mask
            
        except Exception as e:
            print(f"ERROR IN SealSlammersEnv.action_masks(): {e}")
            import traceback
            traceback.print_exc()
            # 在出错时返回一个默认的、结构正确的掩码
            total_mask_size = sum(self.action_space.nvec)
            return np.ones(total_mask_size, dtype=bool)

    def get_info(self):
        try:
            current_player_id = self.game.current_player_turn # 当前轮到行动的玩家

            # Get action masks using the action_masks method
            action_masks_flat = self.action_masks()

            # 返回包含动作掩码和其他调试信息的字典
            # MaskablePPO 会查找 "action_mask" 这个键
            return {
                "action_mask": action_masks_flat,
                "scores": [self.game.scores[0], self.game.scores[1]], # 包含当前分数
                "winner": self.game.winner if hasattr(self.game, 'winner') else None, # 包含胜利者信息
            }
        except Exception as e:
            print(f"ERROR IN SealSlammersEnv.get_info(): {e}")
            import traceback
            traceback.print_exc()
            # 在出错时返回一个默认的、结构正确的掩码，以便让检查通过（但训练会是错误的）
            # 这主要是为了暴露 get_info 内部的错误
            total_actions = self.action_space.nvec[0] * self.action_space.nvec[1] * self.action_space.nvec[2]
            return {
                "action_mask": np.ones(total_actions, dtype=bool),
                "scores": [0, 0], # Default scores
                "winner": None,
                "error_in_get_info": str(e) # 标记错误发生
            }

    # NEW: allow external callback to adjust shaping scale
    def set_shaping_scale(self, scale: float):
        self.shaping_scale = float(max(0.0, min(scale, 1.0)))

    def _calculate_reward(self, initial_scores, initial_hp_states, selected_obj, terminated, current_player_id,
                          collision_happened: bool = False,
                          dist_before: float | None = None,
                          dist_after: float | None = None,
                          alignment_score: float = 0.0,
                          carryover_damage_taken: float | None = None):
        opponent_player_id = 1 - current_player_id
        delta_current = self.game.scores[current_player_id] - initial_scores[current_player_id]
        scale = getattr(self, 'shaping_scale', 1.0)

        # Initialize component values
        ko_points = 0.0
        win_bonus = 0.0
        loss_penalty = 0.0
        invalid_penalty = 0.0
        selection_reward = 0.0
        damage_base_reward = 0.0
        tier_bonus = 0.0
        meaningless_penalty = 0.0
        collision_reward = 0.0
        dist_reward = 0.0
        alignment_reward = 0.0
        position_reward = 0.0
        time_penalty = 0.0

        damage_inflicted = 0.0
        damage_taken = 0.0
        damage_net = 0.0

        if delta_current > 0:
            ko_points = delta_current * KO_POINT_REWARD

        attacker_atk = selected_obj.attack if selected_obj else OBJECT_DEFAULT_ATTACK

        # Compute inflicted / taken from snapshots
        if initial_hp_states:
            # Opponent damage inflicted
            for idx, hp_before in enumerate(initial_hp_states[opponent_player_id]):
                if idx < len(self.game.players_objects[opponent_player_id]):
                    hp_after = self.game.players_objects[opponent_player_id][idx].hp
                    d = hp_before - hp_after
                    if d > 0:
                        damage_inflicted += d
            # immediate self-damage (can be rare); still computed but may be overridden
            for idx, hp_before in enumerate(initial_hp_states[current_player_id]):
                if idx < len(self.game.players_objects[current_player_id]):
                    hp_after = self.game.players_objects[current_player_id][idx].hp
                    d = hp_before - hp_after
                    if d > 0:
                        damage_taken += d
        # Override with cross-turn carryover if provided
        if carryover_damage_taken is not None:
            damage_taken = carryover_damage_taken
        damage_net = damage_inflicted - damage_taken
        any_damage = (damage_inflicted > 1e-6 or damage_taken > 1e-6)

        if not selected_obj:
            invalid_penalty = INVALID_ACTION_PENALTY
        else:
            selection_reward = SELECTION_SHAPING_REWARD * scale

            if any_damage:
                # Semi-zero-sum damage base
                damage_base_reward = (damage_inflicted - DAMAGE_TAKEN_COEFF * damage_taken) * DAMAGE_BASE_PER_HP

                # Tier bonus based ONLY on inflicted (offensive encouragement)
                if damage_inflicted > 0 and attacker_atk > 0:
                    if damage_inflicted >= attacker_atk * 3:
                        tier_bonus = attacker_atk * 1.5 * scale
                    elif damage_inflicted >= attacker_atk * 2:
                        tier_bonus = attacker_atk * 1.0 * scale
                    elif damage_inflicted >= attacker_atk * 1:
                        tier_bonus = attacker_atk * 0.5 * scale

            # Updated meaningless logic
            if delta_current == 0 and damage_inflicted < MEANINGLESS_DAMAGE_THRESHOLD:
                self.consecutive_meaningless_actions += 1
                if self.consecutive_meaningless_actions > MEANINGLESS_GRACE_STEPS:
                    steps_over = self.consecutive_meaningless_actions - MEANINGLESS_GRACE_STEPS - 1
                    base_penalty = min(MEANINGLESS_BASE + steps_over * MEANINGLESS_STEP, MEANINGLESS_MAX)
                    meaningless_penalty = base_penalty * scale
            else:
                self.consecutive_meaningless_actions = 0

            if collision_happened and not any_damage:
                collision_reward = COLLISION_SHAPING_REWARD * scale

            if dist_before is not None and dist_after is not None and dist_after < dist_before:
                delta = dist_before - dist_after
                dist_reward = min(delta / 80.0, DIST_REDUCTION_MAX_SHAPING) * scale

            if alignment_score > 0:
                alignment_reward = min(alignment_score, 1.0) * ALIGNMENT_MAX_SHAPING * scale

        if selected_obj and not terminated:
            position_reward = POSITION_SHAPING_REWARD * scale

        if terminated:
            if self.game.winner == current_player_id:
                win_bonus = WIN_REWARD
            elif self.game.winner == opponent_player_id:
                loss_penalty = -abs(LOSS_PENALTY)
        else:
            time_penalty = TIME_STEP_PENALTY

        reward = (ko_points + win_bonus + loss_penalty + invalid_penalty + selection_reward + damage_base_reward +
                  tier_bonus - meaningless_penalty + collision_reward + dist_reward + alignment_reward +
                  position_reward + time_penalty)

        components = {
            'ko_points': ko_points,
            'win_bonus': win_bonus,
            'loss_penalty': loss_penalty,
            'invalid_penalty': invalid_penalty,
            'selection_reward': selection_reward,
            'damage_inflicted': damage_inflicted,
            'damage_taken': damage_taken,
            'damage_net': damage_net,
            'damage_base': damage_base_reward,
            'tier_bonus': tier_bonus,
            'meaningless_penalty': meaningless_penalty,
            'collision_reward': collision_reward,
            'dist_reward': dist_reward,
            'alignment_reward': alignment_reward,
            'position_reward': position_reward,
            'time_penalty': time_penalty,
            'total_reward': reward
        }

        return reward, components

# --- Game loop and logic (unchanged) ---
    def step(self, action):
        # The 'action' parameter is the action taken by the current_player_id
        # This action will become the 'last_opponent_action' for the *next* player's observation
        
        object_idx, angle_idx, strength_idx = action
        current_player_id = self.game.current_player_turn

        # Reset collision flag for this action
        self.game.enemy_collision_happened_this_step = False

        # ADDED: Check if the agent is attempting to select an object that has already moved this turn.
        attempted_to_select_moved_object = False
        if 0 <= object_idx < len(self.game.players_objects[current_player_id]):
            candidate_obj_for_check = self.game.players_objects[current_player_id][object_idx]
            if candidate_obj_for_check.hp > 0 and candidate_obj_for_check.has_moved_this_turn:
                attempted_to_select_moved_object = True
        
        selected_obj = None
        # Check if the chosen object is valid (belongs to current player and is alive and hasn't moved)
        if 0 <= object_idx < len(self.game.players_objects[current_player_id]):
            candidate_obj = self.game.players_objects[current_player_id][object_idx]
            if candidate_obj.hp > 0 and not candidate_obj.has_moved_this_turn: # Added check for has_moved_this_turn
                selected_obj = candidate_obj
        
        initial_scores = list(self.game.scores) # Copy scores

        # Store initial HP for more detailed reward calculation
        initial_hp_states = []
        for p_objs in self.game.players_objects:
            initial_hp_states.append([obj.hp for obj in p_objs])

        # NEW: precompute nearest opponent distance and alignment before launch
        dist_before = None
        alignment_score = 0.0

        if selected_obj:
            # Find nearest opponent BEFORE applying force
            opponent_objs_alive = [o for o in self.game.players_objects[1 - current_player_id] if o.hp > 0]
            if opponent_objs_alive:
                dists = [math.hypot(selected_obj.x - o.x, selected_obj.y - o.y) for o in opponent_objs_alive]
                nearest_idx = int(np.argmin(dists))
                dist_before = dists[nearest_idx]
                target_opp = opponent_objs_alive[nearest_idx]
            else:
                target_opp = None

        if selected_obj:
            pull_angle_rad = (angle_idx / 72.0) * 2 * math.pi  # 72 个方向
            launch_angle_rad = pull_angle_rad + math.pi  # Launch is opposite to pull

            # Compute alignment_score (0..1) toward nearest opponent at launch time
            if 'target_opp' in locals() and target_opp is not None:
                launch_dir = np.array([math.cos(launch_angle_rad), math.sin(launch_angle_rad)], dtype=float)
                to_opp_vec = np.array([target_opp.x - selected_obj.x, target_opp.y - selected_obj.y], dtype=float)
                to_opp_norm = np.linalg.norm(to_opp_vec)
                if to_opp_norm > 1e-6:
                    to_opp_unit = to_opp_vec / to_opp_norm
                    cos_sim = float(np.clip(np.dot(launch_dir, to_opp_unit), -1.0, 1.0))
                    alignment_score = max(cos_sim, 0.0)  # Only reward when pointing roughly toward opponent

            # ADDED: Set the object's angle based on the RL agent's chosen launch direction.
            selected_obj.angle = launch_angle_rad

            strength_scale = (strength_idx + 1) / 5.0  # 5 档力度: 0..4 -> 0.2..1.0
            actual_strength_magnitude = strength_scale * MAX_LAUNCH_STRENGTH_RL

            launch_vx = math.cos(launch_angle_rad) * actual_strength_magnitude
            launch_vy = math.sin(launch_angle_rad) * actual_strength_magnitude

            if FORCE_MULTIPLIER == 0:
                dx_to_pass_to_apply_force = 0
                dy_to_pass_to_apply_force = 0
            else:
                # Convert desired launch velocity back to pull vector with unified multiplier
                dx_to_pass_to_apply_force = -launch_vx / FORCE_MULTIPLIER
                dy_to_pass_to_apply_force = -launch_vy / FORCE_MULTIPLIER
            selected_obj.apply_force(dx_to_pass_to_apply_force, dy_to_pass_to_apply_force, strength_multiplier=FORCE_MULTIPLIER)
            self.game.action_processing_pending = True
            action_taken_by_current_player = [float(object_idx), float(angle_idx), float(strength_idx)]
        else:
            # Invalid action (e.g., chose KO'd object or bad index)
            # No game object moves, effectively a pass or lost turn.
            # Physics simulation will run but likely nothing happens from player action.
            self.game.action_processing_pending = True # Still need to run simulation loop once.
            action_taken_by_current_player = [float(object_idx), float(angle_idx), float(strength_idx)]

        # Simulate game until objects stop moving or max steps for this turn
        frames_this_step = 0
        MAX_FRAMES_PER_ACTION_STEP = 300 # Max simulation frames per RL step

        # Store the player ID who took this specific action, for damage attribution if needed.
        # self.game.acting_player_this_step = current_player_id (Game class needs this attribute if used)
        # For now, self.game.current_player_turn is assumed to be the one who initiated the action.

        while self.game.action_processing_pending and frames_this_step < MAX_FRAMES_PER_ACTION_STEP:
            still_active = self.game._simulate_frame_physics_and_damage()
            if not still_active: # All objects stopped
                self.game.action_processing_pending = False
            
            frames_this_step += 1
            
            if self.render_mode == "human": # Render intermediate frames of the action
                 self.render()
            
            if self.game.game_over: # If game ends mid-action simulation
                break
        
        self.game.action_processing_pending = False # Ensure it's false after loop

        terminated = self.game.game_over
        
        # Post-simulation HP snapshot BEFORE respawns (for delayed damage_taken)
        post_hp_states = [[obj.hp for obj in p_objs] for p_objs in self.game.players_objects]
        snapshot_prev = self._hp_snapshot_since_last_turn[current_player_id]
        # Ensure length match
        if snapshot_prev is None or len(snapshot_prev) != len(post_hp_states[current_player_id]):
            snapshot_prev = post_hp_states[current_player_id]
        carryover_damage_taken = 0.0
        for prev, now in zip(snapshot_prev, post_hp_states[current_player_id]):
            diff = prev - now
            if diff > 0: carryover_damage_taken += diff

        # Compute post distance
        dist_after = None
        if selected_obj:
            opponent_objs_alive_after = [o for o in self.game.players_objects[1 - current_player_id] if o.hp > 0]
            if opponent_objs_alive_after:
                dists_after = [math.hypot(selected_obj.x - o.x, selected_obj.y - o.y) for o in opponent_objs_alive_after]
                dist_after = min(dists_after)

        # Calculate reward using the new dedicated function (with shaping signals)
        reward, components = self._calculate_reward(
            initial_scores, initial_hp_states, selected_obj, terminated, current_player_id,
            collision_happened=self.game.enemy_collision_happened_this_step,
            dist_before=dist_before, dist_after=dist_after,
            alignment_score=alignment_score,
            carryover_damage_taken=carryover_damage_taken
        )
        
        # ADDED: Apply the specific, additional penalty if an already moved object was selected.
        if attempted_to_select_moved_object:
            reward -= 10.0  # 大幅减少惩罚，从250降到10

        # Update snapshot for current player NOW (before respawns heal) for next turn damage_taken
        self._hp_snapshot_since_last_turn[current_player_id] = post_hp_states[current_player_id]

        # Respawn KO\'d objects before turn progression (if game not over)
        if not terminated:
            for player_list_idx, player_objs in enumerate(self.game.players_objects): # Use enumerate if idx needed
                for obj_to_respawn in player_objs: # Iterate directly over objects
                    if obj_to_respawn.hp <= 0:
                        preferred_x = obj_to_respawn.original_x
                        preferred_y = obj_to_respawn.original_y
                        
                        # Find a non-overlapping position using the Game class method
                        final_x, final_y = self.game._find_non_overlapping_spawn_position(
                            obj_to_respawn, preferred_x, preferred_y
                        )
                        obj_to_respawn.respawn(final_x, final_y) # Pass the found position
            
            # Turn progression logic (from main.py)
            # current_player_id is the player who just took the action.
            # self.game.current_player_turn might have changed if game.next_round() was called by game over.
            # Re-evaluate based on current_player_id who just acted.
            
            # If game ended, self.game.current_player_turn might not be relevant for next action.
            # The turn logic below assumes game is NOT terminated.
            
            player_who_just_moved = current_player_id 
            # Check game status again in case respawn logic or other things change it (unlikely here)
            if not self.game.game_over:
                potential_next_player = 1 - player_who_just_moved
                
                can_potential_next_player_move = self.game.can_player_move(potential_next_player)
                can_player_who_just_moved_still_move = self.game.can_player_move(player_who_just_moved)

                if can_potential_next_player_move:
                    self.game.current_player_turn = potential_next_player
                    # print(f"DEBUG ENV: Turn switched to P{self.game.current_player_turn + 1}")
                elif not can_player_who_just_moved_still_move: # Current player also cannot move
                    # print("DEBUG ENV: Both players cannot move. Starting next round.")
                    self.game.next_round() 
                # Else: current player (player_who_just_moved) can still move,
                # and potential_next_player cannot. So, current player continues.
                # self.game.current_player_turn remains player_who_just_moved.
                # print(f"DEBUG ENV: P{player_who_just_moved+1} continues turn or next round starts.")
        
        # Before getting the observation for the *next* state (which will be for the player whose turn it is *now*),
        # set self.last_opponent_action to the action just taken by 'current_player_id'.
        # This is because 'current_player_id' was the opponent from the perspective of the player
        # who will receive the next observation.
        self.last_opponent_action = action_taken_by_current_player
        
        observation = self._get_obs()
        info = self.get_info() 
        truncated = (frames_this_step >= MAX_FRAMES_PER_ACTION_STEP) and not terminated

        # --- Per-episode component accumulation ---
        if not hasattr(self, '_episode_components'):
            self._episode_components = {k: 0.0 for k in components.keys()}
        for k, v in components.items():
            self._episode_components[k] = self._episode_components.get(k, 0.0) + float(v)
        if terminated or truncated:
            # Provide cumulative components for the finished episode
            info['episode_reward_components'] = self._episode_components.copy()
            # Reset for next episode start (will also be reinitialized on env.reset())
            self._episode_components = {k: 0.0 for k in components.keys()}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.render_mode == "human":
            if self.window_surface is None: # Initialize if not done (e.g. if render_mode changed)
                if not pygame.get_init(): pygame.init()
                if not pygame.font.get_init(): pygame.font.init()
                self.window_surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Seal Slammers RL Environment")
                self.clock = pygame.time.Clock()
                self.game.screen = self.window_surface # Ensure game uses this screen
                self.game.headless_mode = False

            self.game.draw_game_state(self.window_surface) # Game class handles drawing its elements
            pygame.display.flip()
            if self.clock: self.clock.tick(self.metadata['render_fps'])
            return True
        
        elif self.render_mode == "rgb_array":
            # Create a temporary surface for rgb_array rendering if game is headless
            # or if we want a consistent rendering path for rgb_array
            temp_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # Store original game screen settings
            original_game_screen = self.game.screen
            original_headless_mode = self.game.headless_mode

            self.game.screen = temp_surface
            self.game.headless_mode = False # Ensure drawing happens
            if not self.game.font: # Ensure fonts are loaded if game was purely headless
                if not pygame.font.get_init(): pygame.font.init()
                self.game.font = pygame.font.SysFont(None, 30)
                self.game.font_small = pygame.font.SysFont(None, 18)
                self.game.font_hp_atk = pygame.font.SysFont(None, 18)

            self.game.draw_game_state(temp_surface)
            
            rgb_array = pygame.surfarray.array3d(temp_surface)
            
            # Restore original game screen settings
            self.game.screen = original_game_screen
            self.game.headless_mode = original_headless_mode
            
            return np.transpose(rgb_array, (1, 0, 2)) # Convert (width, height, C) to (height, width, C) for gym convention
        
        return None


    def close(self):
        """Close the environment's display resources safely.
        If this is the last human-render environment, fully quit pygame to release all resources.
        """
        # Only decrement if this env was using a human display
        if self.render_mode == 'human':
            if SealSlammersEnv._active_display_envs > 0:
                SealSlammersEnv._active_display_envs -= 1
            # If this was the last active display env, quit pygame fully
            if SealSlammersEnv._active_display_envs == 0 and pygame.get_init():
                try:
                    pygame.quit()
                except Exception as e:
                    print(f"SealSlammersEnv.close(): Exception during pygame.quit(): {e}")
            else:
                # If others still exist, just ensure this env's display surface reference is cleared
                if pygame.display.get_init():
                    try:
                        pygame.display.quit()
                    except Exception:
                        pass
        # Clear references
        self.window_surface = None
        # Prevent double-closing logic on repeated calls
        self.render_mode = None
        return
