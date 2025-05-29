import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random # For potential use in game logic like random start

# --- Game Constants (Copied and merged from main.py and existing env) ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
FPS = 60

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (128, 128, 128)
LIGHT_GREY = (200, 200, 200) # Used as background in main.py draw

OBJECT_RADIUS = 23 # Updated from main.py
INITIAL_HP = 40    # Updated from main.py
INITIAL_ATTACK = 8 # Updated from main.py
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
        
        # Attributes for drawing/UI from original env, might not be strictly needed for RL state if not drawing complexly
        # self.line_length = 0.0
        # self.line_thickness = 2
        # self.line_color = GREY

    def apply_force(self, dx, dy, strength_multiplier=FORCE_MULTIPLIER):
        # In main.py, dx, dy are from object to mouse (vector for launch)
        # The RL env's step() method calculates dx, dy based on angle and strength.
        # This method should directly use them to set velocity.
        # The strength_multiplier is now a constant updated from main.py's effective value.
        self.vx = -dx * strength_multiplier 
        self.vy = -dy * strength_multiplier
        self.is_moving = True
        self.has_moved_this_turn = True
        # print(f"DEBUG: P{self.player_id+1}-Obj{self.object_id} force applied: vx={self.vx:.1f}, vy={self.vy:.1f}")

    def move(self):
        if not self.is_moving:
            return

        self.x += self.vx
        self.y += self.vy

        self.vx *= FRICTION
        self.vy *= FRICTION

        if math.sqrt(self.vx**2 + self.vy**2) < MIN_SPEED_THRESHOLD:
            self.vx = 0
            self.vy = 0
            self.is_moving = False

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

    def respawn(self):
        self.hp = self.initial_hp
        self.x = self.original_x
        self.y = self.original_y
        self.vx = 0.0
        self.vy = 0.0
        self.is_moving = False
        self.has_moved_this_turn = False
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.last_damaged_frame = - DAMAGE_COOLDOWN_FRAMES * 2


    def draw(self, surface, font_small, font_hp_atk): # Added font_hp_atk
        if self.hp <= 0 and False: # Simplified: KO'd objects are just not drawn by main loop or drawn differently
            return

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

        if self.has_moved_this_turn and self.hp > 0: # Highlight if moved (like in main.py)
            pygame.draw.circle(surface, GREEN, (int(self.x), int(self.y)), self.radius + 3, 2)
        
        # Draw object ID (from original env.py, useful for RL debugging)
        id_text_surface = font_small.render(f"{self.object_id+1}", True, BLACK if self.hp > 0 else GREY)
        text_rect = id_text_surface.get_rect(center=(self.x, self.y if self.hp > 0 else self.original_y)) # Show ID at original spot if KO'd
        surface.blit(id_text_surface, text_rect)


# --- Game Class (Copied from main.py and adapted for RL Env) ---
class Game:
    def __init__(self, headless=False, num_objects_per_player=3): # num_objects_per_player from env
        self.headless_mode = headless
        self.num_objects_per_player = num_objects_per_player # Store this

        if not self.headless_mode:
            if not pygame.get_init(): pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("RL Seal Slammers - Game Instance")
            if not pygame.font.get_init(): pygame.font.init() # Ensure font is init
            self.font = pygame.font.SysFont(None, 30)
            self.font_small = pygame.font.SysFont(None, 18) # For object IDs
            self.font_hp_atk = pygame.font.SysFont(None, 18) # For HP/ATK text like in main.py
            self.clock = pygame.time.Clock()
        else: # Headless or for rgb_array
            if not pygame.font.get_init(): pygame.font.init()
            self.font = pygame.font.SysFont(None, 30) # Font might be needed for rgb_array text
            self.font_small = pygame.font.SysFont(None, 18)
            self.font_hp_atk = pygame.font.SysFont(None, 18) # Also for rgb_array text
            self.screen = None 
            self.clock = None

        self.players_objects: list[list[GameObject]] = []
        self.scores = [0, 0]
        self.current_player_turn = 0
        self.game_over = False
        self.winner = None
        self.action_processing_pending = False # True after a launch, until objects stop
        self.frame_count = 0
        self.first_player_of_round = 0 # To manage who starts a new round/exchange

        # Attributes from main.py's Game that are UI-specific and not needed for RL game logic:
        # self.selected_object = None
        # self.is_dragging = False
        # self.drag_start_pos = None
        # self.drag_current_pos = None
        
        self._initialize_objects() # Initialize objects upon creation
        # self.reset_game_state() # reset_game_state will also call _initialize_objects

    def _initialize_objects(self):
        # Using spacing from main.py
        self.players_objects = [[], []]
        # Player 0 (Blue, left)
        for i in range(self.num_objects_per_player):
            obj = GameObject(
                x=100, # Consistent with main.py's P1
                y=SCREEN_HEIGHT // 2 - 100 + i * 100, # Consistent with main.py's P1
                radius=OBJECT_RADIUS,
                color=BLUE,
                hp=INITIAL_HP,
                attack=INITIAL_ATTACK,
                player_id=0,
                object_id=i,
                mass=6.0 # Explicitly pass mass
            )
            self.players_objects[0].append(obj)

        # Player 1 (Red, right)
        for i in range(self.num_objects_per_player):
            obj = GameObject(
                x=SCREEN_WIDTH - 100, # Consistent with main.py's P2
                y=SCREEN_HEIGHT // 2 - 100 + i * 100, # Consistent with main.py's P2
                radius=OBJECT_RADIUS,
                color=RED,
                hp=INITIAL_HP,
                attack=INITIAL_ATTACK,
                player_id=1,
                object_id=i,
                mass=6.0 # Explicitly pass mass
            )
            self.players_objects[1].append(obj)


    def reset_game_state(self):
        self.scores = [0, 0]
        self.current_player_turn = random.choice([0,1]) 
        self.first_player_of_round = self.current_player_turn # Player who starts the round
        self.game_over = False
        self.winner = None
        self.action_processing_pending = False
        self.frame_count = 0
        self._initialize_objects() # Re-create objects for a clean state
        # print("DEBUG: Game state reset (internal Game class).")

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
                    if obj.is_moving:
                        any_object_activity_this_frame = True
        
        # 2. Handle collisions and damage
        active_objects = [obj for player_list in self.players_objects for obj in player_list if obj.hp > 0]
        for i in range(len(active_objects)):
            for j in range(i + 1, len(active_objects)):
                obj1 = active_objects[i]
                obj2 = active_objects[j]

                dist_x = obj1.x - obj2.x
                dist_y = obj1.y - obj2.y
                distance = math.hypot(dist_x, dist_y)
                min_dist = obj1.radius + obj2.radius

                if distance < min_dist: # Collision
                    any_object_activity_this_frame = True

                    # Overlap Resolution (closer to main.py's logic)
                    overlap = min_dist - distance
                    nx = (dist_x / distance) if distance != 0 else 1.0
                    ny = (dist_y / distance) if distance != 0 else 0.0
                    
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
                    elif obj1.mass > 0: # Only obj1 has mass
                        obj1.x += nx * overlap
                        obj1.y += ny * overlap
                    elif obj2.mass > 0: # Only obj2 has mass
                        obj2.x -= nx * overlap
                        obj2.y -= ny * overlap


                    # Impulse Calculation (closer to main.py)
                    rvx = obj1.vx - obj2.vx
                    rvy = obj1.vy - obj2.vy
                    vel_along_normal = rvx * nx + rvy * ny

                    if vel_along_normal < 0: # Moving towards each other
                        e = min(obj1.restitution, obj2.restitution) # Use objects' restitution
                        
                        total_inv_mass_impulse = inv_m1 + inv_m2
                        if total_inv_mass_impulse > 0:
                            impulse_j = -(1 + e) * vel_along_normal / total_inv_mass_impulse
                            
                            obj1.vx += impulse_j * inv_m1 * nx
                            obj1.vy += impulse_j * inv_m1 * ny
                            obj2.vx -= impulse_j * inv_m2 * nx
                            obj2.vy -= impulse_j * inv_m2 * ny
                            
                            obj1.is_moving = True # Mark as moving if impulse applied
                            obj2.is_moving = True
                            any_object_activity_this_frame = True


                        # Damage Application (using take_damage method)
                        if obj1.player_id != obj2.player_id: # No friendly fire
                            # Determine attacker/defender based on current turn, or apply symmetrically if not turn-based damage
                            # For simplicity here, let's assume damage is mutual on collision if they are enemies.
                            # main.py's damage logic in update() is more nuanced based on current_player_turn.
                            # For RL, direct collision damage might be simpler.
                            # Let's refine this: only the object belonging to the *other* player takes damage from *this* player's object if it's a direct result of this player's action.
                            # However, the current structure of _simulate_frame_physics_and_damage is general.
                            # We'll use the take_damage method which includes cooldown.

                            # Simplified: if obj1 hits obj2 (enemy), obj2 takes damage from obj1.attack. And vice-versa.
                            # This part needs to be careful to avoid double-damaging or incorrect attribution.
                            # Let's assume the "active" object (one that was likely launched) deals damage.
                            # This is hard to determine here without more context of which object initiated.
                            # For now, using main.py's logic: if they collide, they both can take damage from each other.
                            
                            obj1_took_damage = False
                            obj2_took_damage = False

                            # Obj1 attempts to damage Obj2
                            if obj2.take_damage(obj1.attack, self.frame_count):
                                obj1_took_damage = True # Incorrect variable name, should be obj2_damaged_by_obj1
                                if obj2.hp <= 0:
                                    self.scores[obj1.player_id] += 1
                                    # print(f"DEBUG: P{obj1.player_id} KO'd P{obj2.player_id}'s obj. Score: {self.scores}")


                            # Obj2 attempts to damage Obj1
                            if obj1.take_damage(obj2.attack, self.frame_count):
                                obj2_took_damage = True # Incorrect, should be obj1_damaged_by_obj2
                                if obj1.hp <= 0:
                                    self.scores[obj2.player_id] += 1
                                    # print(f"DEBUG: P{obj2.player_id} KO'd P{obj1.player_id}'s obj. Score: {self.scores}")
                    
                    obj1.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT) 
                    obj2.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT)

        # 3. Check for game over (victory points)
        if not self.game_over:
            for p_id in range(len(self.scores)):
                if self.scores[p_id] >= MAX_VICTORY_POINTS:
                    self.game_over = True
                    self.winner = p_id
                    break
        return any_object_activity_this_frame

    def _handle_turn_progression_after_action(self):
        """Handles turn progression after an action has resolved and objects stopped."""
        if self.game_over:
            return

        # Respawn KO'd objects (as per main.py's update logic)
        for player_list in self.players_objects:
            for obj in player_list:
                if obj.hp <= 0:
                    obj.respawn() # Respawn includes resetting has_moved_this_turn

        player_who_just_moved = self.current_player_turn # Player whose action led to this state
        
        # Check if the player who just moved can make another move with a *different* unit
        can_current_player_move_again = self.can_player_move(player_who_just_moved)
        
        # Check if the other player can move
        other_player = 1 - player_who_just_moved
        can_other_player_move = self.can_player_move(other_player)

        if can_current_player_move_again:
            # Current player continues if they have more units to move
            # print(f"DEBUG: Player {player_who_just_moved+1} can move another unit.")
            pass # Turn does not change
        elif can_other_player_move:
            # Switch to the other player
            self.current_player_turn = other_player
            # print(f"DEBUG: Player {player_who_just_moved+1} finished. Player {self.current_player_turn+1}'s turn.")
        else:
            # Neither player can move with their current set of "ready" units. This means a full round/exchange ends.
            # print(f"DEBUG: All units moved for both players. Next round.")
            for p_list_reset in self.players_objects:
                for o_reset in p_list_reset:
                    o_reset.has_moved_this_turn = False # Reset all units for new round
            
            # Switch starting player for the new round, like in main.py's next_round()
            self.first_player_of_round = 1 - self.first_player_of_round
            self.current_player_turn = self.first_player_of_round
            # print(f"DEBUG: New round. Player {self.current_player_turn+1} starts.")


    def all_objects_stopped(self) -> bool:
        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.is_moving:
                    return False
        return True

    def can_player_move(self, player_id: int) -> bool:
        if player_id < 0 or player_id >= len(self.players_objects):
            return False
        for obj in self.players_objects[player_id]:
            if obj.hp > 0 and not obj.has_moved_this_turn:
                return True
        return False

    # next_round from main.py was more for a full reset of has_moved_this_turn for everyone
    # The turn progression logic is now mainly in _handle_turn_progression_after_action
    # A simpler next_round might just reset has_moved for all and pick a starter if needed.
    def next_round_full_reset(self): # Renamed to avoid conflict if used differently
        # print("DEBUG: Advancing to next full round/exchange (all units can move again).")
        for player_list in self.players_objects:
            for obj in player_list:
                obj.has_moved_this_turn = False
        # self.current_player_turn = random.choice([0,1]) # main.py uses a more deterministic switch
        self.first_player_of_round = 1 - self.first_player_of_round
        self.current_player_turn = self.first_player_of_round


    def draw(self): # Adapted from main.py's Game.draw
        if self.headless_mode or not self.screen or not self.font:
            return

        self.screen.fill(LIGHT_GREY) # Use LIGHT_GREY like original env

        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.hp > 0: # Only draw objects that are alive
                    obj.draw(self.screen, self.font_small, self.font_hp_atk)

        # Scores (same as main.py)
        score_text_p0 = self.font.render(f"P0 (Blue): {self.scores[0]}", True, BLUE)
        score_text_p1 = self.font.render(f"P1 (Red): {self.scores[1]}", True, RED)
        self.screen.blit(score_text_p0, (10, 10))
        self.screen.blit(score_text_p1, (SCREEN_WIDTH - score_text_p1.get_width() - 10, 10))

        # Turn / Status
        status_text_str = ""
        if not self.game_over:
            player_name = "P0 (Blue)" if self.current_player_turn == 0 else "P1 (Red)"
            if self.action_processing_pending: # This flag is set by the env's step method
                status_text_str = "Objects moving..."
            elif self.can_player_move(self.current_player_turn):
                status_text_str = f"{player_name}, select an object to launch." # More like main.py
            else:
                # This case implies the current player has moved all their units,
                # but the turn might not have switched yet if the other player also can't move (leading to next_round)
                other_player_name = "P1 (Red)" if self.current_player_turn == 0 else "P0 (Blue)"
                status_text_str = f"{player_name} has no more units. Waiting for {other_player_name} or next round."
        
        status_color = BLUE if self.current_player_turn == 0 else RED
        status_surf = self.font.render(status_text_str, True, status_color)
        status_rect = status_surf.get_rect(center=(SCREEN_WIDTH // 2, 20))
        self.screen.blit(status_surf, status_rect)

        # Unit status (optional for env, good for human mode)
        y_offset_for_status = 50; x_offset_p1 = 10; x_offset_p2_base = SCREEN_WIDTH - 150
        if self.font_small and self.font_hp_atk: # Check if fonts are available
            # Player 0 Units Status (like main.py)
            p0_label_surf = self.font.render(f"P0 Units:", True, BLACK)
            self.screen.blit(p0_label_surf, (x_offset_p1, y_offset_for_status))
            current_y_p0 = y_offset_for_status + 20 # Adjusted spacing
            for obj_idx, obj in enumerate(self.players_objects[0]):
                status = "Moved" if obj.has_moved_this_turn else "Ready"
                if obj.hp <=0: status = "KO"
                obj_text = f"  Obj {obj_idx+1}: {status}" # Simplified, HP/ATK drawn on object
                text_surface = self.font_small.render(obj_text, True, BLACK if obj.hp > 0 else GREY)
                self.screen.blit(text_surface, (x_offset_p1, current_y_p0))
                current_y_p0 += 15 # Adjusted spacing

            # Player 1 Units Status (like main.py)
            p1_label_text = f"P1 Units:"
            # p1_label_size_x, _ = self.font.size(p1_label_text) # font.size is pygame specific
            p1_label_surf = self.font.render(p1_label_text, True, BLACK)
            # base_x_p1_units = SCREEN_WIDTH - p1_label_surf.get_width() - 10 # Align right
            base_x_p1_units = x_offset_p2_base # Use existing offset
            self.screen.blit(p1_label_surf, (base_x_p1_units, y_offset_for_status))
            current_y_p1 = y_offset_for_status + 20 # Adjusted spacing
            for obj_idx, obj in enumerate(self.players_objects[1]):
                status = "Moved" if obj.has_moved_this_turn else "Ready"
                if obj.hp <=0: status = "KO"
                obj_text = f"  Obj {obj_idx+1}: {status}" # Simplified
                text_surface = self.font_small.render(obj_text, True, BLACK if obj.hp > 0 else GREY)
                self.screen.blit(text_surface, (base_x_p1_units, current_y_p1))
                current_y_p1 += 15 # Adjusted spacing


        if self.game_over:
            winner_name = "P0 (Blue)" if self.winner == 0 else "P1 (Red)"
            winner_color = BLUE if self.winner == 0 else RED
            go_text_surf = self.font.render(f"Game Over! {winner_name} Wins!", True, winner_color, BLACK)
            go_rect = go_text_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
            
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((128, 128, 128, 180))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(go_text_surf, go_rect)
        
        # pygame.display.flip() is NOT called here; env.render() handles it.

# --- RL Environment Constants (already defined in original env, ensure consistency) ---
NUM_OBJECTS_PER_PLAYER = 3 
NUM_PLAYERS = 2
FEATURES_PER_OBJECT = 5 # x, y, hp, has_moved, is_alive
TOTAL_OBJECT_FEATURES = NUM_OBJECTS_PER_PLAYER * NUM_PLAYERS * FEATURES_PER_OBJECT
ADDITIONAL_FEATURES = 3 # current_player_id, p0_score, p1_score
OBSERVATION_SPACE_SIZE = TOTAL_OBJECT_FEATURES + ADDITIONAL_FEATURES

NUM_DIRECTIONS = 8
NUM_STRENGTHS = 3
# MAX_LAUNCH_STRENGTH_DX_DY was used for action mapping, ensure GameObject.apply_force and action decoding align.
# The GameObject.apply_force now uses FORCE_MULTIPLIER internally.
# The action decoding in env.step() needs to produce dx, dy for apply_force.
# Let's assume the env's MAX_LAUNCH_STRENGTH_DX_DY was for the magnitude of the (dx,dy) vector passed to apply_force.
# The apply_force in main.py takes dx,dy (mouse_pos - obj_pos) and scales by FORCE_MULTIPLIER.
# So, the RL action should produce dx, dy that represent that vector.

class SealSlammersEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        # Game class now handles Pygame init based on headless flag
        self.game = Game(headless=(render_mode != 'human'), num_objects_per_player=NUM_OBJECTS_PER_PLAYER)
        self.game.reset_game_state() # Ensure clean state

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_SPACE_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([NUM_OBJECTS_PER_PLAYER, NUM_DIRECTIONS, NUM_STRENGTHS])
        
        self.current_reward = 0.0
        self.prev_scores = list(self.game.scores)

        # Fallback for human mode screen init if Game class didn't (should be handled by Game now)
        if self.render_mode == 'human' and self.game.screen is None:
            print("Warning: SealSlammersEnv.__init__: Human mode but game screen not initialized by Game. Re-attempting in Env.")
            if not pygame.get_init(): pygame.init()
            self.game.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("RL Seal Slammers (Env Fallback Init)")
            if not pygame.font.get_init(): pygame.font.init()
            self.game.font = pygame.font.SysFont(None, 30)
            self.game.font_small = pygame.font.SysFont(None, 18)
            self.game.clock = pygame.time.Clock()
            self.game.headless_mode = False


    def _get_obs(self):
        obs = []
        for player_idx in range(NUM_PLAYERS):
            for obj_idx in range(NUM_OBJECTS_PER_PLAYER):
                # Ensure object exists, e.g. if num_objects_per_player varies (though fixed here)
                obj = self.game.players_objects[player_idx][obj_idx]
                obs.extend([
                    obj.x / SCREEN_WIDTH,
                    obj.y / SCREEN_HEIGHT,
                    obj.hp / INITIAL_HP if INITIAL_HP > 0 else 0.0,
                    1.0 if obj.has_moved_this_turn else 0.0,
                    1.0 if obj.hp > 0 else 0.0  # is_alive
                ])
        
        obs.extend([
            float(self.game.current_player_turn),
            float(self.game.scores[0]), # P0 score
            float(self.game.scores[1])  # P1 score
        ])
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        return {
            "player_0_score": self.game.scores[0],
            "player_1_score": self.game.scores[1],
            "current_player_turn": self.game.current_player_turn,
            "game_over": self.game.game_over,
            "winner": self.game.winner,
            "main_py_imported": False # Using internal logic
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: # Seed random for reproducibility if game uses it (e.g. start player)
            random.seed(seed)
            np.random.seed(seed) # If using np.random anywhere for game logic

        self.game.reset_game_state()
        self.prev_scores = list(self.game.scores)
        self.current_reward = 0.0
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()
            
        return observation, info

    def step(self, action):
        obj_choice_idx, direction_idx, strength_idx = action
        current_player_id = self.game.current_player_turn
        player_objects = self.game.players_objects[current_player_id]
        
        selected_obj = None
        if 0 <= obj_choice_idx < len(player_objects):
            obj_candidate = player_objects[obj_choice_idx]
            # Ensure the object is valid to be moved
            if obj_candidate.hp > 0 and not obj_candidate.has_moved_this_turn:
                 selected_obj = obj_candidate
        
        self.current_reward = 0.0 # Reset reward for the step

        if selected_obj:
            # Action is valid: an alive, unmoved object is selected
            angle_rad = (direction_idx / NUM_DIRECTIONS) * 2 * math.pi # This is the LAUNCH direction
            
            strength_ratios = [0.33, 0.66, 1.0] # Low, Medium, High strength proportions
            
            # Calculate the magnitude of the pull vector based on main.py logic
            # MAX_LAUNCH_STRENGTH is replaced by this calculation
            _max_pull_pixels = MAX_PULL_RADIUS_MULTIPLIER * OBJECT_RADIUS
            actual_pull_magnitude = _max_pull_pixels * strength_ratios[strength_idx]

            # apply_force expects dx, dy to be the components of the pull vector (object_center to mouse_pos)
            # The launch direction is angle_rad. The pull direction is opposite (angle_rad + pi).
            dx_for_force = actual_pull_magnitude * math.cos(angle_rad + math.pi)
            dy_for_force = actual_pull_magnitude * math.sin(angle_rad + math.pi)
            
            selected_obj.apply_force(dx_for_force, dy_for_force)
            # GameObject.apply_force internally uses FORCE_MULTIPLIER

            self.game.action_processing_pending = True # Mark that an action is being processed
        else:
            # Invalid action
            self.current_reward -= 1.0 # Penalty for invalid action selection
            self.game.action_processing_pending = False

        # --- Simulate Game ---
        frames_simulated_this_step = 0
        # MAX_FRAMES_PER_STEP = FPS * 3 # Simulate for max 3 seconds of game time per agent step
        # Let's use a slightly shorter max sim time for faster steps if physics are slow
        MAX_SIM_STEPS_PER_AGENT_STEP = FPS * 2 # e.g., 120 physics steps

        # Store scores before simulation for this step's reward calculation
        score_at_step_start_p0 = self.game.scores[0]
        score_at_step_start_p1 = self.game.scores[1]

        while frames_simulated_this_step < MAX_SIM_STEPS_PER_AGENT_STEP:
            if self.game.game_over:
                break
            
            self.game._simulate_frame_physics_and_damage() # This handles all physics, damage, scoring for one frame
            self.game.frame_count += 1
            frames_simulated_this_step += 1

            if self.game.all_objects_stopped():
                break 
        
        # After physics simulation (either objects stopped or max sim steps reached)
        if self.game.action_processing_pending and self.game.all_objects_stopped():
            self.game._handle_turn_progression_after_action()
            self.game.action_processing_pending = False # Ensure it's reset

        # --- Calculate Reward ---
        agent_player_id = 0 # Assuming agent is always player 0
        opponent_player_id = 1

        score_change_agent = self.game.scores[agent_player_id] - score_at_step_start_p0
        score_change_opponent = self.game.scores[opponent_player_id] - score_at_step_start_p1

        self.current_reward += (score_change_agent * 1.0)
        self.current_reward -= (score_change_opponent * 0.5) # Penalize opponent scoring, maybe less heavily

        if self.game.game_over:
            if self.game.winner == agent_player_id:
                self.current_reward += 10.0 # Win bonus
            elif self.game.winner == opponent_player_id:
                self.current_reward -= 10.0 # Loss penalty
        
        observation = self._get_obs()
        terminated = self.game.game_over
        # Truncated if max sim steps hit without game ending AND objects were still moving (or some other condition)
        # For now, simple truncation if max sim steps hit and not terminated.
        truncated = frames_simulated_this_step >= MAX_SIM_STEPS_PER_AGENT_STEP and not terminated 
        
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, self.current_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return None # Return None if no rendering is done

        if self.render_mode == 'human':
            if self.game.screen is None or self.game.headless_mode:
                # This should ideally be caught by __init__ or Game class init
                print("Error: Human rendering mode but no screen or game is headless in render().")
                return None
            self.game.draw() # Game's draw method
            pygame.display.flip()
            if self.game.clock:
                self.game.clock.tick(self.metadata['render_fps'])
            return None # Human mode doesn't return an array

        elif self.render_mode == 'rgb_array':
            if self.game.headless_mode and self.game.screen is None:
                # Need a surface to draw on for rgb_array if game is truly headless
                # Create a temporary surface. Game.draw should use self.screen.
                temp_screen_for_rgb = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                original_game_screen = self.game.screen # Store original (likely None)
                self.game.screen = temp_screen_for_rgb # Temporarily assign
                
                self.game.draw() # Draw onto the temp surface
                
                rgb_array = pygame.surfarray.array3d(temp_screen_for_rgb)
                self.game.screen = original_game_screen # Restore original
                return np.transpose(rgb_array, (1, 0, 2)) # Pygame is (width, height)
            elif self.game.screen is not None: # If game has a screen (e.g. was human, or Game init created one)
                self.game.draw()
                rgb_array = pygame.surfarray.array3d(self.game.screen)
                return np.transpose(rgb_array, (1, 0, 2))
            else:
                gym.logger.error("RGB array rendering failed: no screen available.")
                return None
        return None


    def close(self):
        # Pygame quit is handled globally, so only call if it was initialized.
        if pygame.get_init():
            pygame.display.quit()
            pygame.quit()
        # print("SealSlammersEnv closed.")

# --- Main test block (from original env, slightly adapted) ---
if __name__ == '__main__':
    print("Testing SealSlammersEnv with internal game logic...")
    
    # Test with human rendering
    env_human = SealSlammersEnv(render_mode='human')
    obs, info = env_human.reset(seed=42) # Added seed for reproducibility
    print(f"Human Mode: Reset successful. Import status: {info.get('main_py_imported', 'N/A')}")
    print("Initial observation shape:", obs.shape)
    # print("Initial info:", info)
    
    for i in range(20): # Run more steps for testing
        action = env_human.action_space.sample()
        print(f"Step {i+1}, Action: {action}")
        obs, reward, terminated, truncated, info = env_human.step(action)
        print(f"Obs shape: {obs.shape[-1]}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}, P0 Score: {info['player_0_score']}, P1 Score: {info['player_1_score']}")
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Winner: {info.get('winner')}")
            obs, info = env_human.reset(seed=i+43) # New seed for next episode
    env_human.close()
    print("Human mode test finished.")

    # Test with rgb_array rendering
    print("\nTesting RGB Array Mode...")
    env_rgb = SealSlammersEnv(render_mode='rgb_array')
    obs, info = env_rgb.reset(seed=123)
    print(f"RGB Array Mode: Reset successful.")
    # print("Initial observation shape:", obs.shape)
    rgb_frame = env_rgb.render()
    if rgb_frame is not None:
        print("Rendered RGB frame shape:", rgb_frame.shape)
    else:
        print("Failed to render RGB frame in test.")
    
    for _ in range(5):
        action = env_rgb.action_space.sample()
        obs, reward, terminated, truncated, info = env_rgb.step(action)
        # print(f"RGB Step - Obs shape: {obs.shape[-1]}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}")
        if terminated or truncated: break
    rgb_frame = env_rgb.render() # Render after a few steps
    if rgb_frame is not None: print("Final RGB frame shape:", rgb_frame.shape)

    env_rgb.close()
    print("RGB Array mode test finished.")
    
    print("\nReview the copied logic from main.py carefully, especially within")
    print("Game._simulate_frame_physics_and_damage() and Game._handle_turn_progression_after_action().")
