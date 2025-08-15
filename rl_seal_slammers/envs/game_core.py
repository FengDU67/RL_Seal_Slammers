"""
Game core for Seal Slammers: constants, GameObject, and Game classes.
Intended for reuse by multiple RL environments without duplicating logic.
"""
from __future__ import annotations

import pygame
import math
import random
from typing import List

# --- Constants ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (128, 128, 128)
LIGHT_GREY = (200, 200, 200)

OBJECT_RADIUS = 23
OBJECT_DEFAULT_HP = 40
OBJECT_DEFAULT_ATTACK = 8

NUM_OBJECTS_PER_PLAYER = 3
INITIAL_HP = 40
INITIAL_ATTACK = 8

# --- Default HP/Attack Ranges for Randomization ---
DEFAULT_HP_RANGE = (35, 50)
DEFAULT_ATK_RANGE = (7, 10)

MAX_PULL_RADIUS_MULTIPLIER = 4.0

# Compute RL max launch strength to mirror main game's max drag distance * FORCE_MULTIPLIER
FORCE_MULTIPLIER = 0.08 * 4
MAX_LAUNCH_STRENGTH_RL = OBJECT_RADIUS * MAX_PULL_RADIUS_MULTIPLIER * FORCE_MULTIPLIER
FRICTION = 0.98
MIN_SPEED_THRESHOLD = 0.1
ELASTICITY = 1.0

MAX_VICTORY_POINTS = 5
DAMAGE_COOLDOWN_FRAMES = 10

HP_BAR_WIDTH = 40
HP_BAR_HEIGHT = 5


class GameObject:
    def __init__(self, x, y, radius, color, hp, attack, player_id, object_id, mass=6.0):
        self.x = float(x)
        self.y = float(y)
        self.radius = radius
        self.color = color
        self.initial_hp = hp
        self.hp = float(hp)
        self.attack = attack
        self.player_id = player_id
        self.object_id = object_id

        self.vx = 0.0
        self.vy = 0.0
        self.is_moving = False
        self.has_moved_this_turn = False
        self.original_x = float(x)
        self.original_y = float(y)
        self.mass = mass
        self.restitution = ELASTICITY
        self.moment_of_inertia = 0.5 * self.mass * (self.radius ** 2)

        self.angle = 0.0
        self.angular_velocity = 0.0
        self.last_damaged_frame = - DAMAGE_COOLDOWN_FRAMES * 2
        self.damage_intake_cooldown_frames = DAMAGE_COOLDOWN_FRAMES

    def apply_force(self, dx, dy, strength_multiplier=0.08 * 4):
        self.vx = -dx * strength_multiplier
        self.vy = -dy * strength_multiplier
        self.is_moving = True
        self.has_moved_this_turn = True

    def move(self):
        if self.is_moving:
            self.x += self.vx
            self.y += self.vy
            self.vx *= FRICTION
            self.vy *= FRICTION
            if abs(self.vx) < MIN_SPEED_THRESHOLD and abs(self.vy) < MIN_SPEED_THRESHOLD:
                self.vx = 0
                self.vy = 0
                self.is_moving = False
        self.angle += self.angular_velocity
        self.angular_velocity *= 0.97

    def check_boundary_collision(self, screen_width, screen_height):
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx *= -self.restitution
        elif self.x + self.radius > screen_width:
            self.x = screen_width - self.radius
            self.vx *= -self.restitution
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy *= -self.restitution
        elif self.y + self.radius > screen_height:
            self.y = screen_height - self.radius
            self.vy *= -self.restitution

    def take_damage(self, amount, current_game_frame):
        if current_game_frame - self.last_damaged_frame < self.damage_intake_cooldown_frames:
            return False
        self.hp -= amount
        self.last_damaged_frame = current_game_frame
        if self.hp < 0:
            self.hp = 0
        return True

    def respawn(self, new_x=None, new_y=None):
        self.hp = self.initial_hp
        self.x = new_x if new_x is not None else self.original_x
        self.y = new_y if new_y is not None else self.original_y
        self.vx = 0.0
        self.vy = 0.0
        self.is_moving = False
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.last_damaged_frame = - DAMAGE_COOLDOWN_FRAMES * 2

    def draw(self, surface, font_small, font_hp_atk):
        if self.hp <= 0:
            return
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        if self.hp > 0:
            hp_text_surface = font_hp_atk.render(f"HP: {self.hp:.0f}", True, BLACK)
            atk_text_surface = font_hp_atk.render(f"ATK: {self.attack}", True, BLACK)
            spacing = 5
            total_text_block_width = hp_text_surface.get_width() + spacing + atk_text_surface.get_width()
            hp_start_x = self.x - total_text_block_width / 2
            atk_start_x = hp_start_x + hp_text_surface.get_width() + spacing
            text_y_pos = self.y + self.radius + 5
            surface.blit(hp_text_surface, (hp_start_x, text_y_pos))
            surface.blit(atk_text_surface, (atk_start_x, text_y_pos))
            line_end_x = self.x + self.radius * math.cos(self.angle)
            line_end_y = self.y + self.radius * math.sin(self.angle)
            pygame.draw.line(surface, BLACK, (self.x, self.y), (line_end_x, line_end_y), 2)
        if self.has_moved_this_turn:
            pygame.draw.circle(surface, GREEN, (int(self.x), int(self.y)), self.radius + 3, 2)
        id_text_surface = font_small.render(f"{self.object_id+1}", True, BLACK if self.hp > 0 else GREY)
        text_rect = id_text_surface.get_rect(center=(self.x, self.y if self.hp > 0 else self.original_y))
        surface.blit(id_text_surface, text_rect)


class Game:
    def __init__(self, headless=False, num_objects_per_player=NUM_OBJECTS_PER_PLAYER):
        if not pygame.get_init() and not headless:
            pygame.init()
        if not pygame.font.get_init() and not headless:
            pygame.font.init()
        self.headless_mode = headless
        self.screen = None
        if not self.headless_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Seal Slammers - Game Instance")
            self.font = pygame.font.SysFont(None, 30)
            self.font_small = pygame.font.SysFont(None, 18)
            self.font_hp_atk = pygame.font.SysFont(None, 18)
            self.clock = pygame.time.Clock()
        else:
            if not pygame.get_init():
                pygame.init()
            if not pygame.font.get_init():
                pygame.font.init()
            self.font = pygame.font.SysFont(None, 30)
            self.font_small = pygame.font.SysFont(None, 18)
            self.font_hp_atk = pygame.font.SysFont(None, 18)
            self.screen = None
            self.clock = None
        self.num_objects_per_player = num_objects_per_player
        self.players_objects: List[List[GameObject]] = []
        self.scores = [0, 0]
        self.current_player_turn = 0
        self.game_over = False
        self.winner = None
        self.action_processing_pending = False
        self.frame_count = 0
        self.first_player_of_round = 0
        self._RESPAWN_SEARCH_STEP_DISTANCE_FACTOR = 0.75
        self._RESPAWN_MAX_SPIRAL_LAYERS = 8
        self._RESPAWN_POINTS_PER_LAYER = 8
        self.enemy_collision_happened_this_step = False
        self.reset_game_state(p0_hp=INITIAL_HP, p0_atk=INITIAL_ATTACK, p1_hp=INITIAL_HP, p1_atk=INITIAL_ATTACK)

    def reset_game_state(self, p0_hp, p0_atk, p1_hp, p1_atk):
        self.players_objects = [[], []]
        self.scores = [0, 0]
        self.current_player_turn = 0
        self.game_over = False
        self.winner = None
        self.action_processing_pending = False
        x_padding_p0 = 50 + OBJECT_RADIUS
        for i in range(self.num_objects_per_player):
            x = x_padding_p0
            y = SCREEN_HEIGHT // (self.num_objects_per_player + 1) * (i + 1)
            obj = GameObject(x, y, OBJECT_RADIUS, BLUE, player_id=0, object_id=i, hp=p0_hp, attack=p0_atk)
            obj.angle = 0
            self.players_objects[0].append(obj)
        x_padding_p1 = SCREEN_WIDTH - 50 - OBJECT_RADIUS
        for i in range(self.num_objects_per_player):
            x = x_padding_p1
            y = SCREEN_HEIGHT // (self.num_objects_per_player + 1) * (i + 1)
            obj = GameObject(x, y, OBJECT_RADIUS, RED, player_id=1, object_id=i, hp=p1_hp, attack=p1_atk)
            obj.angle = math.pi
            self.players_objects[1].append(obj)
        for p_objs in self.players_objects:
            for obj_k in p_objs:
                obj_k.has_moved_this_turn = False

    def all_objects_stopped(self):
        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.is_moving:
                    return False
        return True

    def can_player_move(self, player_id: int) -> bool:
        for obj in self.players_objects[player_id]:
            if obj.hp > 0 and not obj.has_moved_this_turn:
                return True
        return False

    def object_has_moved_this_turn(self, player_id: int, object_idx_in_player_list: int) -> bool:
        if 0 <= player_id < len(self.players_objects) and 0 <= object_idx_in_player_list < len(self.players_objects[player_id]):
            return self.players_objects[player_id][object_idx_in_player_list].has_moved_this_turn
        return True

    def next_round(self):
        self.first_player_of_round = 1 - self.first_player_of_round
        self.current_player_turn = self.first_player_of_round
        for player_objs in self.players_objects:
            for obj in player_objs:
                obj.has_moved_this_turn = False

    def _check_overlap_at_pos(self, x, y, radius, object_to_exclude=None):
        for player_list in self.players_objects:
            for other_obj in player_list:
                if other_obj is object_to_exclude or other_obj.hp <= 0:
                    continue
                dist_sq = (x - other_obj.x)**2 + (y - other_obj.y)**2
                min_dist_sq = (radius + other_obj.radius)**2
                if dist_sq < min_dist_sq - 1e-6:
                    return True
        return False

    def _find_non_overlapping_spawn_position(self, object_being_respawned, preferred_x, preferred_y):
        obj_radius = object_being_respawned.radius
        if not self._check_overlap_at_pos(preferred_x, preferred_y, obj_radius, object_being_respawned):
            return preferred_x, preferred_y
        search_step_distance = obj_radius * self._RESPAWN_SEARCH_STEP_DISTANCE_FACTOR
        for layer in range(1, self._RESPAWN_MAX_SPIRAL_LAYERS + 1):
            current_search_dist = layer * search_step_distance
            for i in range(self._RESPAWN_POINTS_PER_LAYER):
                angle_offset = (math.pi / self._RESPAWN_POINTS_PER_LAYER) * (layer % 2)
                angle = (2 * math.pi / self._RESPAWN_POINTS_PER_LAYER) * i + angle_offset
                candidate_x = preferred_x + current_search_dist * math.cos(angle)
                candidate_y = preferred_y + current_search_dist * math.sin(angle)
                candidate_x = max(obj_radius, min(candidate_x, SCREEN_WIDTH - obj_radius))
                candidate_y = max(obj_radius, min(candidate_y, SCREEN_HEIGHT - obj_radius))
                if not self._check_overlap_at_pos(candidate_x, candidate_y, obj_radius, object_being_respawned):
                    return candidate_x, candidate_y
        return preferred_x, preferred_y

    def _simulate_frame_physics_and_damage(self):
        if self.game_over:
            return False
        self.frame_count += 1
        any_object_activity_this_frame = False
        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.is_moving:
                    obj.move()
                    obj.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT)
                    if obj.is_moving:
                        any_object_activity_this_frame = True
        active_objects = [obj for player_list in self.players_objects for obj in player_list if obj.hp > 0]
        for i in range(len(active_objects)):
            for j in range(i + 1, len(active_objects)):
                obj1 = active_objects[i]
                obj2 = active_objects[j]
                dist_x = obj1.x - obj2.x
                dist_y = obj1.y - obj2.y
                distance = math.hypot(dist_x, dist_y)
                if distance == 0:
                    obj1.x += random.uniform(-0.1, 0.1)
                    obj1.y += random.uniform(-0.1, 0.1)
                    dist_x = obj1.x - obj2.x
                    dist_y = obj1.y - obj2.y
                    distance = math.hypot(dist_x, dist_y)
                    if distance == 0:
                        continue
                min_dist = obj1.radius + obj2.radius
                if distance < min_dist:
                    any_object_activity_this_frame = True
                    if obj1.player_id != obj2.player_id:
                        self.enemy_collision_happened_this_step = True
                    overlap = min_dist - distance
                    nx = dist_x / distance if distance != 0 else 1.0
                    ny = dist_y / distance if distance != 0 else 0.0
                    inv_m1 = 1.0 / obj1.mass if obj1.mass > 0 else 0.0
                    inv_m2 = 1.0 / obj2.mass if obj2.mass > 0 else 0.0
                    total_inv_mass_correction = inv_m1 + inv_m2
                    if total_inv_mass_correction > 0:
                        correction_factor_obj1 = inv_m1 / total_inv_mass_correction
                        correction_factor_obj2 = inv_m2 / total_inv_mass_correction
                        obj1.x += nx * overlap * correction_factor_obj1
                        obj1.y += ny * overlap * correction_factor_obj1
                        obj2.x -= nx * overlap * correction_factor_obj2
                        obj2.y -= ny * overlap * correction_factor_obj2
                    elif obj1.mass > 0:
                        obj1.x += nx * overlap
                        obj1.y += ny * overlap
                    elif obj2.mass > 0:
                        obj2.x -= nx * overlap
                        obj2.y -= ny * overlap
                    rvx = obj1.vx - obj2.vx
                    rvy = obj1.vy - obj2.vy
                    vel_along_normal = rvx * nx + rvy * ny
                    if vel_along_normal < 0:
                        e = min(obj1.restitution, obj2.restitution)
                        total_inv_mass_impulse = inv_m1 + inv_m2
                        if total_inv_mass_impulse > 0:
                            impulse_j = -(1 + e) * vel_along_normal / total_inv_mass_impulse
                            obj1.vx += impulse_j * inv_m1 * nx
                            obj1.vy += impulse_j * inv_m1 * ny
                            obj2.vx -= impulse_j * inv_m2 * nx
                            obj2.vy -= impulse_j * inv_m2 * ny
                            if abs(obj1.vx) > 0.01 or abs(obj1.vy) > 0.01:
                                obj1.is_moving = True
                            if abs(obj2.vx) > 0.01 or abs(obj2.vy) > 0.01:
                                obj2.is_moving = True
                            if obj1.player_id != obj2.player_id:
                                attacker_obj, defender_obj = None, None
                                if obj1.player_id == self.current_player_turn and obj2.player_id != self.current_player_turn:
                                    attacker_obj, defender_obj = obj1, obj2
                                elif obj2.player_id == self.current_player_turn and obj1.player_id != self.current_player_turn:
                                    attacker_obj, defender_obj = obj2, obj1
                                if attacker_obj and defender_obj:
                                    damaged = defender_obj.take_damage(attacker_obj.attack, self.frame_count)
                                    if damaged and defender_obj.hp <= 0:
                                        scoring_player_id = attacker_obj.player_id
                                        self.scores[scoring_player_id] += 1
                                        if self.scores[scoring_player_id] >= MAX_VICTORY_POINTS:
                                            self.game_over = True
                                            self.winner = scoring_player_id
                                            return False
                    obj1.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT)
                    obj2.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT)
        if not any_object_activity_this_frame:
            return False
        return True

    def _reset_round_for_scoring(self):
        pass

    def draw_game_state(self, surface):
        if self.headless_mode and surface is None:
            return
        surface.fill(LIGHT_GREY)
        for player_objs in self.players_objects:
            for obj in player_objs:
                obj.draw(surface, self.font_small, self.font_hp_atk)
        score_text = f"P1 (Blue): {self.scores[0]}  |  P2 (Red): {self.scores[1]}"
        score_surface = self.font.render(score_text, True, BLACK)
        surface.blit(score_surface, (SCREEN_WIDTH // 2 - score_surface.get_width() // 2, 10))
        turn_text = f"Turn: Player {self.current_player_turn + 1}"
        turn_color = BLUE if self.current_player_turn == 0 else RED
        turn_surface = self.font.render(turn_text, True, turn_color)
        surface.blit(turn_surface, (SCREEN_WIDTH // 2 - turn_surface.get_width() // 2, 40))
        can_move = self.can_player_move(self.current_player_turn)
        can_move_text = "Can Move" if can_move else "No Moves Left"
        can_move_color = GREEN if can_move else GREY
        can_move_surface = self.font.render(can_move_text, True, can_move_color)
        surface.blit(can_move_surface, (SCREEN_WIDTH // 2 - can_move_surface.get_width() // 2, 70))
        if self.game_over:
            winner_text = f"Player {self.winner + 1} Wins!" if self.winner is not None else "Game Over - Draw?"
            game_over_message = f"GAME OVER! {winner_text}"
            game_over_surface = self.font.render(game_over_message, True, GREEN if self.winner is not None else BLACK)
            text_rect = game_over_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            bg_rect = text_rect.inflate(20, 20)
            pygame.draw.rect(surface, LIGHT_GREY, bg_rect)
            pygame.draw.rect(surface, BLACK, bg_rect, 2)
            surface.blit(game_over_surface, text_rect)
