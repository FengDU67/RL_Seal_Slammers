"""
连续动作空间版本的Seal Slammers环境
使用Box(3)动作空间: [unit_selection, angle, strength]
- unit_selection: [0, 1] 映射到 [0, num_objects_per_player-1]
- angle: [0, 1] 映射到 [0, 2π]
- strength: [0, 1] 直接使用
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random

# 定义常量（从原环境复制）
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
OBJECT_RADIUS = 23
DEFAULT_HP_RANGE = (35, 50)
DEFAULT_ATK_RANGE = (7, 10)
NUM_OBJECTS_PER_PLAYER = 3
MAX_LAUNCH_STRENGTH_RL = 20.0
FORCE_MULTIPLIER = 0.08 * 4
FRICTION = 0.98
MIN_SPEED_THRESHOLD = 0.1
ELASTICITY = 1.0
MAX_PULL_RADIUS_MULTIPLIER = 4.0
MAX_VICTORY_POINTS = 5
DAMAGE_COOLDOWN_FRAMES = 10
HP_BAR_WIDTH = 40
HP_BAR_HEIGHT = 5

# 颜色常量
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (128, 128, 128)
LIGHT_GREY = (200, 200, 200)

# 从原环境导入类
from rl_seal_slammers.envs.seal_slammers_env import GameObject, Game

class SealSlammersEnvContinuous(gym.Env):
    """
    连续动作空间版本的Seal Slammers环境
    动作空间: Box(3) - [unit_selection, angle, strength]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, num_objects_per_player=NUM_OBJECTS_PER_PLAYER,
                 p0_hp_fixed=None, p0_atk_fixed=None,
                 p1_hp_fixed=None, p1_atk_fixed=None,
                 hp_range=DEFAULT_HP_RANGE,
                 atk_range=DEFAULT_ATK_RANGE):
        
        super().__init__()
        
        self.render_mode = render_mode
        self.num_objects_per_player = num_objects_per_player
        
        # HP/ATK configuration
        self.p0_hp_fixed = p0_hp_fixed
        self.p0_atk_fixed = p0_atk_fixed
        self.p1_hp_fixed = p1_hp_fixed
        self.p1_atk_fixed = p1_atk_fixed
        self.hp_range = hp_range
        self.atk_range = atk_range
        
        self.last_opponent_action = [-1.0, -1.0, -1.0]
        self.consecutive_meaningless_actions = 0
        
        # 连续动作空间: [unit_selection, angle, strength]
        # unit_selection: [0, 1] -> 映射到单位索引
        # angle: [0, 1] -> 映射到 [0, 2π]
        # strength: [0, 1] -> 直接使用
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # 观察空间保持不变
        obs_dim = (num_objects_per_player * 2 * 5) + 1 + 2 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 初始化pygame（如果需要渲染）
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Seal Slammers - Continuous Actions")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        
        # 游戏实例
        self.game = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game:
            self.game.close()
        
        # 初始化游戏
        self.game = Game(
            headless=(self.render_mode != "human" and self.render_mode != "rgb_array"),
            num_objects_per_player=self.num_objects_per_player
        )
        
        # 设置HP和攻击力
        import random
        p0_hp = self.p0_hp_fixed if self.p0_hp_fixed is not None else random.uniform(self.hp_range[0], self.hp_range[1])
        p0_atk = self.p0_atk_fixed if self.p0_atk_fixed is not None else random.uniform(self.atk_range[0], self.atk_range[1])
        p1_hp = self.p1_hp_fixed if self.p1_hp_fixed is not None else random.uniform(self.hp_range[0], self.hp_range[1])
        p1_atk = self.p1_atk_fixed if self.p1_atk_fixed is not None else random.uniform(self.atk_range[0], self.atk_range[1])
        
        self.game.reset_game_state(p0_hp=p0_hp, p0_atk=p0_atk, p1_hp=p1_hp, p1_atk=p1_atk)
        
        self.last_opponent_action = [-1.0, -1.0, -1.0]
        self.consecutive_meaningless_actions = 0
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """
        处理连续动作并实现mask机制
        action: [unit_selection, angle, strength] 均在 [0, 1] 范围内
        """
        # 解析连续动作
        unit_selection_raw, angle_raw, strength_raw = action
        
        current_player_id = self.game.current_player_turn
        
        # 智能单位选择：自动将连续值映射到有效单位
        available_units = []
        player_objects = self.game.players_objects[current_player_id]
        for i, obj in enumerate(player_objects):
            if obj.hp > 0 and not obj.has_moved_this_turn:
                available_units.append(i)
        
        if not available_units:
            # 没有可用单位时的处理
            selected_obj = None
            object_idx = 0
            angle_rad = 0
            strength = 0
            attempted_to_select_moved_object = False
            is_invalid_action = True
        else:
            # 智能单位映射：将 [0, 1] 平滑映射到可用单位
            # 使用更平滑的映射函数，避免边界效应
            normalized_unit_selection = max(0.0, min(0.999, unit_selection_raw))  # 确保在有效范围内
            unit_idx = int(normalized_unit_selection * len(available_units))
            object_idx = available_units[unit_idx]
            selected_obj = player_objects[object_idx]
            
            # 角度映射：将 [0, 1] 映射到 [0, 2π]
            angle_rad = angle_raw * 2 * math.pi
            
            # 力度映射：限制最小力度，避免无意义的微弱攻击
            min_strength = 0.1  # 最小力度阈值
            strength = max(min_strength, strength_raw)
            
            attempted_to_select_moved_object = False
            is_invalid_action = False
        
        # 记录初始状态用于奖励计算
        initial_scores = list(self.game.scores)
        initial_hp_states = []
        for p_objs in self.game.players_objects:
            initial_hp_states.append([obj.hp for obj in p_objs])
        
        # 执行动作
        if selected_obj:
            # 计算发射角度（与拉拽方向相反）
            launch_angle_rad = angle_rad + math.pi
            selected_obj.angle = launch_angle_rad
            
            # 计算实际力度
            actual_strength_magnitude = strength * MAX_LAUNCH_STRENGTH_RL
            
            # 计算速度分量
            launch_vx = math.cos(launch_angle_rad) * actual_strength_magnitude
            launch_vy = math.sin(launch_angle_rad) * actual_strength_magnitude
            
            # 计算需要传递给apply_force的参数
            if FORCE_MULTIPLIER == 0:
                dx_to_pass = 0
                dy_to_pass = 0
            else:
                dx_to_pass = -launch_vx / FORCE_MULTIPLIER
                dy_to_pass = -launch_vy / FORCE_MULTIPLIER
            
            selected_obj.apply_force(dx_to_pass, dy_to_pass)
            self.game.action_processing_pending = True
            
            # 记录动作（用于观察）
            action_taken = [float(object_idx), angle_raw, strength_raw]
        else:
            # 无效动作
            self.game.action_processing_pending = True
            action_taken = [0.0, angle_raw, strength_raw]
        
        # 模拟物理过程
        frames_this_step = 0
        MAX_FRAMES_PER_ACTION_STEP = 300
        
        while self.game.action_processing_pending and frames_this_step < MAX_FRAMES_PER_ACTION_STEP:
            still_active = self.game._simulate_frame_physics_and_damage()
            if not still_active:
                self.game.action_processing_pending = False
            
            frames_this_step += 1
            
            if self.render_mode == "human":
                self.render()
            
            if self.game.game_over:
                break
        
        self.game.action_processing_pending = False
        
        terminated = self.game.game_over
        
        # 计算奖励
        reward = self._calculate_reward(
            initial_scores, initial_hp_states, selected_obj, 
            terminated, current_player_id
        )
        
        # 额外惩罚（如果尝试选择已移动的对象）
        if attempted_to_select_moved_object:
            reward -= 10.0
        
        # 重生被击败的对象
        if not terminated:
            for player_list_idx, player_objs in enumerate(self.game.players_objects):
                for obj_to_respawn in player_objs:
                    if obj_to_respawn.hp <= 0:
                        preferred_x = obj_to_respawn.original_x
                        preferred_y = obj_to_respawn.original_y
                        
                        final_x, final_y = self.game._find_non_overlapping_spawn_position(
                            obj_to_respawn, preferred_x, preferred_y
                        )
                        
                        obj_to_respawn.respawn(final_x, final_y)
        
        # 更新回合
        if not self.game.action_processing_pending:
            self._handle_turn_progression()
        
        # 更新对手动作记录
        if self.game.current_player_turn != current_player_id:
            self.last_opponent_action = action_taken
        
        obs = self._get_obs()
        info = {
            'scores': self.game.scores,
            'current_player': self.game.current_player_turn,
            'action_taken': action_taken,
            'selected_object': object_idx if selected_obj else None
        }
        
        return obs, reward, terminated, False, info
    
    def _get_obs(self):
        """获取观察状态（与原环境相同）"""
        state = []
        
        # 计算屏幕中心用于相对位置计算
        center_x = SCREEN_WIDTH / 2
        center_y = SCREEN_HEIGHT / 2
        
        # 对象特征（使用相对位置）
        for player_objs in self.game.players_objects:
            for obj in player_objs:
                # 相对位置（归一化到 [-1, 1]）
                rel_x = (obj.x - center_x) / (SCREEN_WIDTH / 2)
                rel_y = (obj.y - center_y) / (SCREEN_HEIGHT / 2)
                
                # 归一化HP和攻击力
                norm_hp = obj.hp / 100.0
                norm_attack = obj.attack / 20.0
                
                # 移动状态
                has_moved = 1.0 if obj.has_moved_this_turn else 0.0
                
                state.extend([rel_x, rel_y, norm_hp, norm_attack, has_moved])
        
        # 当前玩家回合
        state.append(float(self.game.current_player_turn))
        
        # 归一化分数
        state.extend([self.game.scores[0] / 10.0, self.game.scores[1] / 10.0])
        
        # 上一个对手动作（已经是归一化的）
        state.extend(self.last_opponent_action)
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self, initial_scores, initial_hp_states, selected_obj, 
                         terminated, current_player_id):
        """计算奖励（与原环境相同的逻辑）"""
        reward = 0.0
        
        # 分数奖励
        score_diff = self.game.scores[current_player_id] - initial_scores[current_player_id]
        reward += score_diff * 100.0
        
        # 伤害奖励
        for player_idx, player_objs in enumerate(self.game.players_objects):
            for obj_idx, obj in enumerate(player_objs):
                initial_hp = initial_hp_states[player_idx][obj_idx]
                current_hp = obj.hp
                hp_lost = initial_hp - current_hp
                
                if hp_lost > 0:
                    if player_idx == current_player_id:
                        # 己方受伤，惩罚
                        reward -= hp_lost * 2.0
                    else:
                        # 对手受伤，奖励
                        reward += hp_lost * 2.0
        
        # 无效动作惩罚
        if selected_obj is None:
            reward -= 50.0
        
        # 终局奖励
        if terminated:
            if self.game.scores[current_player_id] > self.game.scores[1 - current_player_id]:
                reward += 200.0  # 胜利奖励
            else:
                reward -= 200.0  # 失败惩罚
        
        return reward
    
    def _handle_turn_progression(self):
        """处理回合进展逻辑（从原环境复制）"""
        player_who_just_moved = self.game.current_player_turn
        potential_next_player = 1 - player_who_just_moved

        can_potential_next_player_move = self.game.can_player_move(potential_next_player)
        can_player_who_just_moved_still_move = self.game.can_player_move(player_who_just_moved)

        if can_potential_next_player_move:
            self.game.current_player_turn = potential_next_player
        elif can_player_who_just_moved_still_move:
            # 当前玩家继续行动
            pass
        else:
            # 双方都不能移动，开始新回合
            self.game.next_round()
    
    def render(self):
        """渲染游戏画面"""
        if self.render_mode == "human" and self.screen:
            self.game.render(self.screen)
            pygame.display.flip()
            self.clock.tick(60)
    
    def close(self):
        """关闭环境"""
        if self.game and hasattr(self.game, 'close'):
            self.game.close()
        if self.screen:
            pygame.quit()
