import os # Added
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random # Ensure random is imported

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
LIGHT_GREY = (200, 200, 200) # Used as background in main.py draw

OBJECT_RADIUS = 23 # Updated from main.py
OBJECT_DEFAULT_HP = 40    # Default HP if not randomized/specified
OBJECT_DEFAULT_ATTACK = 8 # Default Attack if not randomized/specified

NUM_OBJECTS_PER_PLAYER = 3 # Added definition
INITIAL_HP = 40
INITIAL_ATTACK = 8

# --- Default HP/Attack Ranges for Randomization ---
DEFAULT_HP_RANGE = (35, 50)
DEFAULT_ATK_RANGE = (7, 10)

MAX_LAUNCH_STRENGTH_RL = 20.0 # Added for RL action scaling
# MAX_LAUNCH_STRENGTH = 200 # This is more related to RL action scaling
FORCE_MULTIPLIER = 0.08 * 4 # Adjusted to match main.py's effective multiplier
FRICTION = 0.98 
MIN_SPEED_THRESHOLD = 0.1 
ELASTICITY = 1.0 # Updated from main.py's restitution
MAX_PULL_RADIUS_MULTIPLIER = 4.0 # Added from main.py logic

MAX_VICTORY_POINTS = 5
DAMAGE_COOLDOWN_FRAMES = 10 # From main.py

HP_BAR_WIDTH = 40
HP_BAR_HEIGHT = 5

# --- GameObject Class (Copied from main.py and adapted) ---
class GameObject:
    def __init__(self, x, y, radius, color, hp, attack, player_id, object_id, mass=6.0): # Updated mass from main.py
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
        self.restitution = ELASTICITY # Match main.py's attribute name for clarity within GameObject

        # Attributes from main.py's GameObject
        self.angle = 0.0 # For potential drawing or physics extensions
        self.angular_velocity = 0.0 # For potential physics extensions
        self.last_damaged_frame = - DAMAGE_COOLDOWN_FRAMES * 2 # Initialize to allow immediate damage
        self.damage_intake_cooldown_frames = DAMAGE_COOLDOWN_FRAMES
        

    def apply_force(self, dx, dy, strength_multiplier=FORCE_MULTIPLIER):
        self.vx = -dx * strength_multiplier 
        self.vy = -dy * strength_multiplier
        self.is_moving = True
        self.has_moved_this_turn = True # Consistent with main.py

    def move(self):
        if self.is_moving:
            self.x += self.vx
            self.y += self.vy

            self.vx *= FRICTION 
            self.vy *= FRICTION
            
            if math.hypot(self.vx, self.vy) < MIN_SPEED_THRESHOLD:
                self.vx = 0
                self.vy = 0
                self.is_moving = False
                self.angular_velocity = 0 # Stop spinning if it was spinning

        # Update angle from angular velocity (for spinning effect) - this remains
        self.angle += self.angular_velocity
        self.angular_velocity *= 0.97 # Angular friction/damping (TODO: Use a constant like ANGULAR_FRICTION)

    def check_boundary_collision(self, screen_width, screen_height):
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx *= -self.restitution # Use self.restitution
        elif self.x + self.radius > screen_width:
            self.x = screen_width - self.radius
            self.vx *= -self.restitution # Use self.restitution

        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy *= -self.restitution # Use self.restitution
        elif self.y + self.radius > screen_height:
            self.y = screen_height - self.radius
            self.vy *= -self.restitution # Use self.restitution
            # Dampen if hitting bottom slowly (from main.py like logic, though main.py doesn't have this exact detail)
            # This specific damping was in original env, keeping it.
            if abs(self.vy) < MIN_SPEED_THRESHOLD * 2 and self.y + self.radius >= screen_height -1:
                 self.vy = 0
    
    def take_damage(self, amount, current_game_frame):
        if current_game_frame - self.last_damaged_frame < self.damage_intake_cooldown_frames:
            return False # In cooldown
        self.hp -= amount
        self.last_damaged_frame = current_game_frame
        if self.hp < 0:
            self.hp = 0
        return True # Damage applied

    def respawn(self, new_x=None, new_y=None): # Modified to accept new_x, new_y
        self.hp = self.initial_hp
        self.x = new_x if new_x is not None else self.original_x # Use new position if provided
        self.y = new_y if new_y is not None else self.original_y # Use new position if provided
        self.vx = 0.0
        self.vy = 0.0
        self.is_moving = False
        # self.has_moved_this_turn = False # Reset by next_round or turn logic, not individually here
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.last_damaged_frame = - DAMAGE_COOLDOWN_FRAMES * 2


    def draw(self, surface, font_small, font_hp_atk): # Added font_hp_atk
        if self.hp <= 0: # If HP is 0 or less, do not draw the object
            return       # This makes it disappear

        # Draw circle (same as main.py)
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw HP and ATK text (like in main.py)
        if self.hp > 0 : # Only draw if alive
            hp_text_surface = font_hp_atk.render(f"HP: {self.hp:.0f}", True, BLACK)
            atk_text_surface = font_hp_atk.render(f"ATK: {self.attack}", True, BLACK)
            spacing = 5
            total_text_block_width = hp_text_surface.get_width() + spacing + atk_text_surface.get_width()
            hp_start_x = self.x - total_text_block_width / 2
            atk_start_x = hp_start_x + hp_text_surface.get_width() + spacing
            text_y_pos = self.y + self.radius + 5
            surface.blit(hp_text_surface, (hp_start_x, text_y_pos))
            surface.blit(atk_text_surface, (atk_start_x, text_y_pos))

            # Draw direction indicator (like in main.py)
            # Angle for RL agent might be set differently or not used for this visual
            # For now, let's assume self.angle might be updated by RL logic if needed for visuals
            line_end_x = self.x + self.radius * math.cos(self.angle)
            line_end_y = self.y + self.radius * math.sin(self.angle)
            pygame.draw.line(surface, BLACK, (self.x, self.y), (line_end_x, line_end_y), 2)

        if self.has_moved_this_turn: # Highlight if moved (like in main.py) - Removed 'and self.hp > 0' for consistency
            pygame.draw.circle(surface, GREEN, (int(self.x), int(self.y)), self.radius + 3, 2)
        
        # Draw object ID (from original env.py, useful for RL debugging)
        id_text_surface = font_small.render(f"{self.object_id+1}", True, BLACK if self.hp > 0 else GREY)
        text_rect = id_text_surface.get_rect(center=(self.x, self.y if self.hp > 0 else self.original_y)) # Show ID at original spot if KO'd
        surface.blit(id_text_surface, text_rect)


# --- Game Class (Copied from main.py and adapted for RL Env) ---
class Game:
    def __init__(
        self, headless=False, num_objects_per_player=NUM_OBJECTS_PER_PLAYER): # Removed HP/ATK params
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
            self.font_small = pygame.font.SysFont(None, 18) # For object IDs
            self.font_hp_atk = pygame.font.SysFont(None, 18) # For HP/ATK text like in main.py
            self.clock = pygame.time.Clock()
        else: # Headless or for rgb_array
            # Initialize fonts even in headless mode for potential rgb_array rendering
            if not pygame.get_init(): pygame.init() # Pygame needs to be init for fonts
            if not pygame.font.get_init(): pygame.font.init()
            self.font = pygame.font.SysFont(None, 30) 
            self.font_small = pygame.font.SysFont(None, 18)
            self.font_hp_atk = pygame.font.SysFont(None, 18)
            self.screen = None 
            self.clock = None # Clock not typically used in headless RL steps unless for FPS matching

        self.num_objects_per_player = num_objects_per_player
        # Removed self.pX_initial_hp and self.pX_initial_atk attributes

        self.players_objects: list[list[GameObject]] = []
        self.scores = [0, 0]
        self.current_player_turn = 0
        self.game_over = False
        self.winner = None
        self.action_processing_pending = False # True after a launch, until objects stop
        self.frame_count = 0
        self.first_player_of_round = 0 # To manage who starts a new round/exchange
        
        # Constants for respawn logic
        self._RESPAWN_SEARCH_STEP_DISTANCE_FACTOR = 0.75 # Factor of object radius
        self._RESPAWN_MAX_SPIRAL_LAYERS = 8
        self._RESPAWN_POINTS_PER_LAYER = 8

        # Call reset_game_state with default HP/Attack values
        self.reset_game_state(p0_hp=INITIAL_HP, p0_atk=INITIAL_ATTACK, 
                              p1_hp=INITIAL_HP, p1_atk=INITIAL_ATTACK)

    def reset_game_state(self, p0_hp, p0_atk, p1_hp, p1_atk): # Added HP/ATK params
        self.players_objects = [[], []]
        self.scores = [0, 0]
        self.current_player_turn = 0  # Player 0 starts
        self.game_over = False
        self.winner = None
        self.action_processing_pending = False # True if an action has been taken and physics sim is ongoing

        # Initial positions for player 0 (left side, vertical)
        x_padding_p0 = 50 + OBJECT_RADIUS
        for i in range(self.num_objects_per_player):
            x = x_padding_p0
            y = SCREEN_HEIGHT // (self.num_objects_per_player + 1) * (i + 1)
            # Create GameObject without 'angle' in constructor
            obj = GameObject(x, y, OBJECT_RADIUS, BLUE, player_id=0, object_id=i,
                             hp=p0_hp, attack=p0_atk)
            obj.angle = 0 # Facing right
            self.players_objects[0].append(obj)

        # Initial positions for player 1 (right side, vertical)
        x_padding_p1 = SCREEN_WIDTH - 50 - OBJECT_RADIUS
        for i in range(self.num_objects_per_player):
            x = x_padding_p1
            y = SCREEN_HEIGHT // (self.num_objects_per_player + 1) * (i + 1)
            # Create GameObject without 'angle' in constructor
            obj = GameObject(x, y, OBJECT_RADIUS, RED, player_id=1, object_id=i,
                             hp=p1_hp, attack=p1_atk)
            obj.angle = math.pi # Facing left
            self.players_objects[1].append(obj)
        
        for p_objs in self.players_objects:
            for obj_k in p_objs:
                obj_k.has_moved_this_turn = False
    

    def all_objects_stopped(self): # Added from main.py
        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.is_moving:
                    return False
        return True

    def can_player_move(self, player_id: int) -> bool: # Added from main.py
        """Checks if the specified player has any units that can still move."""
        for obj in self.players_objects[player_id]:
            if obj.hp > 0 and not obj.has_moved_this_turn:
                return True
        return False

    def object_has_moved_this_turn(self, player_id: int, object_idx_in_player_list: int) -> bool:
        if 0 <= player_id < len(self.players_objects) and \
           0 <= object_idx_in_player_list < len(self.players_objects[player_id]):
            return self.players_objects[player_id][object_idx_in_player_list].has_moved_this_turn
        # If indices are out of bounds, or object doesn't exist,
        # consider it as "cannot be moved" or "has effectively moved / is not available".
        # This prevents errors if play.py tries to check an invalid index.
        return True


    def next_round(self): # Added from main.py
        # print("DEBUG ENV: Round ended, switching first player for next round.")
        self.first_player_of_round = 1 - self.first_player_of_round
        self.current_player_turn = self.first_player_of_round
        for player_objs in self.players_objects:
            for obj in player_objs:
                obj.has_moved_this_turn = False # Reset all units' move status
        # print(f"DEBUG ENV: Next round. Player {self.current_player_turn + 1} starts.")


    def _check_overlap_at_pos(self, x, y, radius, object_to_exclude=None):
        """Checks if an object at (x,y) with radius would overlap with other active objects."""
        for player_list in self.players_objects:
            for other_obj in player_list:
                if other_obj is object_to_exclude or other_obj.hp <= 0:
                    continue # Don't check against self or KO'd objects

                dist_sq = (x - other_obj.x)**2 + (y - other_obj.y)**2
                min_dist_sq = (radius + other_obj.radius)**2
                if dist_sq < min_dist_sq - 1e-6: # Added tolerance for floating point
                    return True # Overlap detected
        return False # No overlap

    def _find_non_overlapping_spawn_position(self, object_being_respawned, preferred_x, preferred_y):
        """
        Finds a non-overlapping position for the object_being_respawned,
        starting from preferred_x, preferred_y and searching outwards if needed.
        """
        obj_radius = object_being_respawned.radius

        # Try preferred position first
        if not self._check_overlap_at_pos(preferred_x, preferred_y, obj_radius, object_being_respawned):
            return preferred_x, preferred_y

        # Spiral search outwards
        search_step_distance = obj_radius * self._RESPAWN_SEARCH_STEP_DISTANCE_FACTOR

        for layer in range(1, self._RESPAWN_MAX_SPIRAL_LAYERS + 1):
            current_search_dist = layer * search_step_distance
            for i in range(self._RESPAWN_POINTS_PER_LAYER):
                # Add a slight angle offset for each layer to ensure points don't always align
                angle_offset = (math.pi / self._RESPAWN_POINTS_PER_LAYER) * (layer % 2) 
                angle = (2 * math.pi / self._RESPAWN_POINTS_PER_LAYER) * i + angle_offset
                
                candidate_x = preferred_x + current_search_dist * math.cos(angle)
                candidate_y = preferred_y + current_search_dist * math.sin(angle)

                # Clamp to screen boundaries
                candidate_x = max(obj_radius, min(candidate_x, SCREEN_WIDTH - obj_radius))
                candidate_y = max(obj_radius, min(candidate_y, SCREEN_HEIGHT - obj_radius))

                if not self._check_overlap_at_pos(candidate_x, candidate_y, obj_radius, object_being_respawned):
                    return candidate_x, candidate_y
        
        # Fallback: return preferred position even if it overlaps (should be rare)
        # print(f"WARN: Could not find non-overlapping spawn for P{object_being_respawned.player_id}-Obj{object_being_respawned.object_id}, using preferred.")
        return preferred_x, preferred_y


    def _simulate_frame_physics_and_damage(self):
        """
        Simulates one frame of game physics: movement, boundary collisions,
        object-object collisions, damage, scoring, and game over checks.
        Returns True if any object was moving or a collision occurred.
        # This is derived from main.py's Game.update() method's core logic.
        # Also increments self.frame_count
        """
        if self.game_over:
            return False
        
        self.frame_count +=1
        any_object_activity_this_frame = False

        # 1. Move all objects
        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.is_moving:
                    obj.move()
                    obj.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT)
                    if obj.is_moving: # Check again as move() might set it to False
                        any_object_activity_this_frame = True
        
        # 2. Handle collisions and damage (Ported from main.py Game.update)
        active_objects = [obj for player_list in self.players_objects for obj in player_list if obj.hp > 0]
        for i in range(len(active_objects)):
            for j in range(i + 1, len(active_objects)):
                obj1 = active_objects[i]
                obj2 = active_objects[j]

                dist_x = obj1.x - obj2.x
                dist_y = obj1.y - obj2.y
                distance = math.hypot(dist_x, dist_y)

                if distance == 0: # Avoid division by zero
                    obj1.x += random.uniform(-0.1, 0.1)
                    obj1.y += random.uniform(-0.1, 0.1)
                    dist_x = obj1.x - obj2.x; dist_y = obj1.y - obj2.y
                    distance = math.hypot(dist_x, dist_y)
                    if distance == 0: continue

                min_dist = obj1.radius + obj2.radius
                if distance < min_dist:
                    any_object_activity_this_frame = True # Collision occurred
                    # --- Overlap Resolution (from main.py) ---
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
                    elif obj1.mass > 0: obj1.x += nx * overlap; obj1.y += ny * overlap
                    elif obj2.mass > 0: obj2.x -= nx * overlap; obj2.y -= ny * overlap
                    
                    # --- Impulse Calculation (from main.py) ---
                    rvx = obj1.vx - obj2.vx
                    rvy = obj1.vy - obj2.vy
                    vel_along_normal = rvx * nx + rvy * ny

                    if vel_along_normal < 0: # Objects are moving towards each other
                        e = min(obj1.restitution, obj2.restitution)
                        total_inv_mass_impulse = inv_m1 + inv_m2
                        if total_inv_mass_impulse > 0:
                            impulse_j = -(1 + e) * vel_along_normal / total_inv_mass_impulse
                            obj1.vx += impulse_j * inv_m1 * nx
                            obj1.vy += impulse_j * inv_m1 * ny
                            obj2.vx -= impulse_j * inv_m2 * nx
                            obj2.vy -= impulse_j * inv_m2 * ny

                            if abs(obj1.vx) > 0.01 or abs(obj1.vy) > 0.01: obj1.is_moving = True
                            if abs(obj2.vx) > 0.01 or abs(obj2.vy) > 0.01: obj2.is_moving = True

                            # --- Damage Application (from main.py) ---
                            if obj1.player_id != obj2.player_id:
                                # Determine attacker (current player's object involved in collision)
                                # and defender (opponent's object)
                                # For RL, the "current_player_turn" is the one whose action initiated movement.
                                # If obj1 or obj2 was the one launched by current_player_turn, or
                                # became active due to that launch.
                                # Simpler: if one is current_player_turn's team and other is not.
                                # Let's assume damage is dealt by the object of the player whose turn it *was*
                                # when the action was taken. This is stored in env.game.current_player_turn
                                # (which is `current_player_id` in the step function before this simulation)
                                # This part is tricky. main.py uses self.current_player_turn.
                                # For the env, the "active" player for damage dealing should be the one
                                # whose action is currently being processed.
                                # Let's assume the `self.current_player_turn` in Game class reflects who *initiated* the action.
                                
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
                                        # print(f"DEBUG ENV: Player {scoring_player_id + 1} scores. Scores: {self.scores}")
                                        if self.scores[scoring_player_id] >= MAX_VICTORY_POINTS:
                                            self.game_over = True
                                            self.winner = scoring_player_id
                                            # print(f"DEBUG ENV: Player {self.winner + 1} wins!")
                                            return False # Game ended, stop simulation

                    obj1.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT)
                    obj2.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT)

        # Removed old scoring logic based on team wipe. Scoring is per KO now.
        # Game over is checked per KO.

        if not any_object_activity_this_frame:
            return False # All objects are stationary

        return True # Objects were active or collision occurred

    def _reset_round_for_scoring(self):
        # This method is no longer used in the same way as main.py.
        # main.py respawns all KO'd units at the end of an action sequence,
        # and starts a new round if no player can move.
        # Kept for now in case it's needed for a different game mode, but current logic won't call it.
        pass

    def draw_game_state(self, surface):
        if self.headless_mode and surface is None: # Added check for surface is None for rgb_array case if game is headless
            # If truly headless and no surface provided (e.g. for pure logic steps), do nothing.
            # However, rgb_array mode in Env will provide a surface.
            return

        surface.fill(LIGHT_GREY)  # Background color

        # Draw all game objects
        for player_objs in self.players_objects:
            for obj in player_objs:
                # GameObject.draw handles its own drawing logic, including for KO'd objects
                obj.draw(surface, self.font_small, self.font_hp_atk)

        # Draw scores
        score_text = f"P1 (Blue): {self.scores[0]}  |  P2 (Red): {self.scores[1]}"
        score_surface = self.font.render(score_text, True, BLACK)
        surface.blit(score_surface, (SCREEN_WIDTH // 2 - score_surface.get_width() // 2, 10))

        # Draw current player turn
        turn_text = f"Turn: Player {self.current_player_turn + 1}"
        turn_color = BLUE if self.current_player_turn == 0 else RED
        turn_surface = self.font.render(turn_text, True, turn_color)
        surface.blit(turn_surface, (SCREEN_WIDTH // 2 - turn_surface.get_width() // 2, 40))
        
        # Draw "Can Move" indicator for current player
        can_move = self.can_player_move(self.current_player_turn)
        can_move_text = "Can Move" if can_move else "No Moves Left"
        can_move_color = GREEN if can_move else GREY
        can_move_surface = self.font.render(can_move_text, True, can_move_color)
        surface.blit(can_move_surface, (SCREEN_WIDTH // 2 - can_move_surface.get_width() // 2, 70))


        # Draw Game Over message if applicable
        if self.game_over:
            winner_text = ""
            if self.winner is not None:
                winner_text = f"Player {self.winner + 1} Wins!"
            else: # Should not happen if game_over is True and MAX_VICTORY_POINTS logic is correct
                winner_text = "Game Over - Draw?" 
            
            game_over_message = f"GAME OVER! {winner_text}"
            game_over_surface = self.font.render(game_over_message, True, GREEN if self.winner is not None else BLACK)
            text_rect = game_over_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            
            # Simple background for game over text
            bg_rect = text_rect.inflate(20, 20)
            pygame.draw.rect(surface, LIGHT_GREY, bg_rect)
            pygame.draw.rect(surface, BLACK, bg_rect, 2)
            surface.blit(game_over_surface, text_rect)

# --- SealSlammersEnv Class (Gym Environment) ---
class SealSlammersEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode=None, num_objects_per_player=NUM_OBJECTS_PER_PLAYER,
                 p0_hp_fixed=None, p0_atk_fixed=None, # For fixed HP/ATK e.g. in play.py
                 p1_hp_fixed=None, p1_atk_fixed=None,
                 hp_range=DEFAULT_HP_RANGE,         # For random HP/ATK in training
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

        # Action space:
        # action[0]: object_to_move (0 to num_objects_per_player - 1)
        # action[1]: angle_idx (0 to 71 for 72 directions)
        # action[2]: strength_idx (0 to 4 for 5 levels)
        self.action_space = spaces.MultiDiscrete([
            num_objects_per_player, 
            72,                     
            5                       
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
        #   - Opponent\'s chosen object index (1 feature)
        #   - Opponent\'s chosen angle index (1 feature)
        #   - Opponent\'s chosen strength index (1 feature)
        # Total features = (num_objects_per_player * 2 * 5) + 1 + 2 + 3
        obs_dim = (num_objects_per_player * 2 * 5) + 1 + 2 + 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # REMOVED return statement from __init__

    def _get_obs(self):
        # Flattened state: all objects\' states + current player turn + scores
        # For each object: x, y, hp, attack, has_moved (5 features) - ALL RAW VALUES
        # Global: current_player_turn (1), score_p0 (1), score_p1 (1) - ALL RAW VALUES
        # Last opponent action: object index, angle index, strength index (3 features) - ALL RAW VALUES
        state = []
        
        for player_objs in self.game.players_objects:
            for obj in player_objs:
                state.extend([
                    obj.x,
                    obj.y,
                    obj.hp,
                    obj.attack,
                    1.0 if obj.has_moved_this_turn else 0.0,
                ])

        # Current player turn (0 or 1)
        state.append(float(self.game.current_player_turn))

        # Raw Scores
        state.extend([float(s) for s in self.game.scores]) # Ensure scores are float

        # Add the last opponent's action to the observation
        state.extend([float(a) for a in self.last_opponent_action])

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

        # --- Observation Construction ---
        observation = self._get_obs()
        # 在 reset 方法中调用 get_info() 以确保初始状态也包含动作掩码
        info = self.get_info() 
        
        if self.render_mode == "human":
            self.render()
        return observation, info

    def get_info(self):
        try:
            current_player_id = self.game.current_player_turn # 当前轮到行动的玩家

            # 1. 创建对象选择掩码 (object_mask)
            # 初始化为所有对象都不可选
            object_mask_list = [False] * self.num_objects_per_player
            
            # 只有当当前玩家有棋子可以移动时，才计算有效的对象掩码
            if self.game.can_player_move(current_player_id):
                player_objects = self.game.players_objects[current_player_id]
                for i, obj in enumerate(player_objects):
                    # 对象可选的条件是：存活(hp > 0) 且 本回合尚未移动
                    if obj.hp > 0 and not self.game.object_has_moved_this_turn(current_player_id, i):
                        object_mask_list[i] = True
            
            object_mask_np = np.array(object_mask_list, dtype=bool)

            # 2. 创建角度掩码 (angle_mask) - 通常所有角度都可选
            # self.action_space.nvec[1] 对应的是动作空间中角度的数量 (例如 72)
            angle_mask_np = np.full(self.action_space.nvec[1], True, dtype=bool)

            # 3. 创建力度掩码 (strength_mask) - 通常所有力度都可选
            # self.action_space.nvec[2] 对应的是动作空间中力度的数量 (例如 5)
            strength_mask_np = np.full(self.action_space.nvec[2], True, dtype=bool)
            
            # 将三个掩码组合成一个元组
            action_masks_tuple = (object_mask_np, angle_mask_np, strength_mask_np)

            # 返回包含动作掩码和其他调试信息的字典
            # MaskablePPO 会查找 "action_mask" 这个键
            return {
                "action_mask": action_masks_tuple,
                "scores": [self.game.scores[0], self.game.scores[1]], # 包含当前分数
                "winner": self.game.winner if hasattr(self.game, 'winner') else None, # 包含胜利者信息
                # 您可以根据需要添加其他调试信息
                # "can_player_move": self.game.can_player_move(current_player_id),
                # "current_player_turn_in_info": current_player_id
            }
        except Exception as e:
            print(f"ERROR IN SealSlammersEnv.get_info(): {e}")
            import traceback
            traceback.print_exc()
            # 在出错时返回一个默认的、结构正确的掩码，以便让检查通过（但训练会是错误的）
            # 这主要是为了暴露 get_info 内部的错误
            default_object_mask = np.full(self.num_objects_per_player, True, dtype=bool)
            default_angle_mask = np.full(self.action_space.nvec[1], True, dtype=bool)
            default_strength_mask = np.full(self.action_space.nvec[2], True, dtype=bool)
            return {
                "action_mask": (default_object_mask, default_angle_mask, default_strength_mask),
                "scores": [0, 0], # Default scores
                "winner": None,
                "error_in_get_info": str(e) # 标记错误发生
            }

    def _calculate_reward(self, initial_scores, initial_hp_states, selected_obj, terminated, current_player_id):
        """
        Calculates the reward for the current step.
        """
        reward = 0.0
        score_diff_p_current = self.game.scores[current_player_id] - initial_scores[current_player_id]
        opponent_player_id = 1 - current_player_id
        initial_opponent_score = initial_scores[opponent_player_id]
        current_opponent_score = self.game.scores[opponent_player_id]

        # Reward for increasing own score (typically by KOing an opponent)
        reward += score_diff_p_current * 50.0

        # # Penalty for opponent increasing their score (e.g. if self-KO or complex interaction)
        # reward -= (current_opponent_score - initial_opponent_score) * 50.0

        if not selected_obj: # Penalty for invalid action (e.g., chose KO\'d object, out of bounds)
            reward -= 1000.0
            # No need to increment consecutive_meaningless_actions here, as this is a different kind of bad action.
            # Or, we could consider it a "super" meaningless action. For now, keeping separate.
        else: # Valid object was selected
            # --- Reward for damaging an opponent (Tiered based on damage relative to attacker's ATK) ---
            any_damage_dealt_to_opponent = False
            total_damage_dealt_to_opponent_hp = 0 # Tracks total HP damage dealt
            
            # Get the attack power of the selected object (attacker)
            attacker_atk = selected_obj.attack if selected_obj else OBJECT_DEFAULT_ATTACK # Fallback if selected_obj is None (should not happen here)

            if initial_hp_states: 
                for obj_idx, hp_before_action in enumerate(initial_hp_states[opponent_player_id]):
                    if obj_idx < len(self.game.players_objects[opponent_player_id]): 
                        obj_after_action = self.game.players_objects[opponent_player_id][obj_idx]
                        hp_after_action = obj_after_action.hp
                        damage_dealt_to_this_obj = hp_before_action - hp_after_action
                        
                        if damage_dealt_to_this_obj > 0:
                            total_damage_dealt_to_opponent_hp += damage_dealt_to_this_obj
                            any_damage_dealt_to_opponent = True
                            
                            # Tiered reward based on damage dealt relative to attacker's ATK
                            # This approximates rewarding "hitting multiple enemies" by rewarding high total damage in one go.
                            # If actual_strength_magnitude was used, it might be a proxy for "hitting hard"
                            # For now, let's use a simpler multiplier on total HP damage.
                            # The idea of "hitting multiple enemies" is better captured by total damage dealt
                            # across all enemies in this single action.

            # Base reward for any damage
            if any_damage_dealt_to_opponent:
                reward += total_damage_dealt_to_opponent_hp * 1.0 # Base reward for HP damage

                # Tiered bonus based on total damage dealt in this action vs attacker's ATK
                # This is a proxy for "hitting multiple units hard" or "a very effective hit"
                if attacker_atk > 0: # Avoid division by zero
                    damage_multiplier_tiers = 0
                    if total_damage_dealt_to_opponent_hp >= attacker_atk * 3: # Approx. 3 units hit effectively or one super hit
                        damage_multiplier_tiers = 3 
                        reward += total_damage_dealt_to_opponent_hp * 27.0 # Strongest bonus
                    elif total_damage_dealt_to_opponent_hp >= attacker_atk * 2: # Approx. 2 units hit effectively
                        damage_multiplier_tiers = 2
                        reward += total_damage_dealt_to_opponent_hp * 9.0  # Medium bonus
                    elif total_damage_dealt_to_opponent_hp >= attacker_atk * 1: # Approx. 1 unit hit effectively
                        damage_multiplier_tiers = 1
                        reward += total_damage_dealt_to_opponent_hp * 3.0   # Smallest bonus
            
            # --- Penalty for "meaningless" valid action & Progressive Penalty---
            if score_diff_p_current == 0 and \
               (current_opponent_score - initial_opponent_score) == 0 and \
               not any_damage_dealt_to_opponent:
                self.consecutive_meaningless_actions += 1
                # Base penalty + progressive penalty
                reward -= (10.0 + (self.consecutive_meaningless_actions -1) * 10.0) 
            else:
                # If the action was meaningful, reset the counter
                self.consecutive_meaningless_actions = 0

        # --- Strategic Positioning Reward (Proximity to Opponents) ---
        if selected_obj and not terminated : # Only if a valid action was taken and game not over
            current_player_active_objects = [obj for obj in self.game.players_objects[current_player_id] if obj.hp > 0]
            opponent_active_objects = [obj for obj in self.game.players_objects[opponent_player_id] if obj.hp > 0]

            if current_player_active_objects and opponent_active_objects:
                total_min_dist_to_opponent = 0
                num_active_player_units = 0

                for p_obj in current_player_active_objects:
                    min_dist_for_this_obj = float('inf')
                    for o_obj in opponent_active_objects:
                        dist = math.hypot(p_obj.x - o_obj.x, p_obj.y - o_obj.y)
                        min_dist_for_this_obj = min(min_dist_for_this_obj, dist)
                    
                    if min_dist_for_this_obj != float('inf'):
                        total_min_dist_to_opponent += min_dist_for_this_obj
                        num_active_player_units +=1
                
                if num_active_player_units > 0:
                    avg_min_dist = total_min_dist_to_opponent / num_active_player_units
                    # Reward for being closer. Max screen diagonal is ~sqrt(1000^2 + 600^2) ~= 1166
                    # We want higher reward for smaller distance.
                    # Let's use a scale factor and subtract from a base to make it positive.
                    # Example: Max reward of 2.0 for this component.
                    # (1.0 - (avg_min_dist / SCREEN_WIDTH)) ensures value is between 0 and 1 (approx)
                    # Adjust scaling factor as needed.
                    proximity_reward = (1.0 - (avg_min_dist / (SCREEN_WIDTH * 0.5))) * 0.5 
                    reward += max(0, proximity_reward) # Ensure it doesn't go negative if avg_min_dist is very large

        if terminated:
            if self.game.winner == current_player_id:
                reward += 500.0
            elif self.game.winner == opponent_player_id:
                reward -= 500.0
        else:
            # --- Time-based penalty (if game not terminated) ---
            reward -= 0.5 # Small penalty per step to encourage faster wins

        return reward

    def step(self, action):
        # The 'action' parameter is the action taken by the current_player_id
        # This action will become the 'last_opponent_action' for the *next* player's observation
        
        object_idx, angle_idx, strength_idx = action
        current_player_id = self.game.current_player_turn

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

        if selected_obj:
            pull_angle_rad = (angle_idx / 72.0) * 2 * math.pi  # Angle of the "pull" action (NOW 72 directions)
            launch_angle_rad = pull_angle_rad + math.pi  # Launch is opposite to pull

            # ADDED: Set the object's angle based on the RL agent's chosen launch direction.
            # This ensures the object faces the correct direction at the moment of launch by an agent.
            # This angle will persist during movement due to the change in GameObject.move().
            selected_obj.angle = launch_angle_rad

            strength_scale = (strength_idx + 1) / 5.0  # 0 to 4 -> 0.2 to 1.0
            actual_strength_magnitude = strength_scale * MAX_LAUNCH_STRENGTH_RL # This is launch velocity magnitude

            # launch_vx, launch_vy is the desired velocity of the object in the launch direction
            launch_vx = math.cos(launch_angle_rad) * actual_strength_magnitude
            launch_vy = math.sin(launch_angle_rad) * actual_strength_magnitude
            
            # GameObject.apply_force expects dx, dy to be "pull vector components"
            # To achieve launch_vx, launch_vy, the effective "pull vector" (dx_param, dy_param)
            # passed to apply_force should be such that:
            # launch_vx = -dx_param * FORCE_MULTIPLIER  (since apply_force does self.vx = -dx * multiplier)
            # launch_vy = -dy_param * FORCE_MULTIPLIER
            # So, dx_param = -launch_vx / FORCE_MULTIPLIER
            # and dy_param = -launch_vy / FORCE_MULTIPLIER
            # These dx_param, dy_param will be in the original pull direction.
            if FORCE_MULTIPLIER == 0: # Avoid division by zero
                dx_to_pass_to_apply_force = 0
                dy_to_pass_to_apply_force = 0
            else:
                dx_to_pass_to_apply_force = -launch_vx / FORCE_MULTIPLIER
                dy_to_pass_to_apply_force = -launch_vy / FORCE_MULTIPLIER
            
            selected_obj.apply_force(dx_to_pass_to_apply_force, dy_to_pass_to_apply_force)
            self.game.action_processing_pending = True
            # Record the valid action taken by the current player
            action_taken_by_current_player = [float(object_idx), float(angle_idx), float(strength_idx)]
        else:
            # Invalid action (e.g., chose KO'd object or bad index)
            # No game object moves, effectively a pass or lost turn.
            # Physics simulation will run but likely nothing happens from player action.
            self.game.action_processing_pending = True # Still need to run simulation loop once.
            # Record the invalid action attempted by the current player
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
        
        self.game.action_processing_pending = False # Ensure it\'s false after loop

        terminated = self.game.game_over
        
        # Calculate reward using the new dedicated function
        reward = self._calculate_reward(initial_scores, initial_hp_states, selected_obj, terminated, current_player_id)
        
        # ADDED: Apply the specific, additional penalty if an already moved object was selected.
        # This stacks on top of the -1000 penalty from _calculate_reward (because selected_obj would be None).
        if attempted_to_select_moved_object:
            reward -= 250.0  # Example value for the explicit, independent penalty. Tune as needed.
            
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
        # 在 step 方法中调用 get_info() 以获取新状态的动作掩码
        info = self.get_info() 
        truncated = (frames_this_step >= MAX_FRAMES_PER_ACTION_STEP) and not terminated

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
        if self.window_surface is not None:
            pygame.display.quit()
            self.window_surface = None
        # Pygame quit should ideally be called when the application is fully closing,
        # not necessarily when one env instance is closed, if other pygame stuff is running.
        # However, if this env is the main user of pygame display, it's fine.
        # For safety, let's assume we only quit display here.
        # If pygame.get_init() and no other modules depend on it, pygame.quit() could be called.
        # pygame.quit() # This uninitializes all pygame modules.
        pass # Avoid global pygame.quit() for now.

# Example of how to register (optional, can be done elsewhere)
# from gymnasium.envs.registration import register
# register(
#     id='SealSlammers-v0',
#     entry_point='__main__:SealSlammersEnv', # Or path to this file:module
#     max_episode_steps=2000, # Max steps per episode if not handled by termination
# )
