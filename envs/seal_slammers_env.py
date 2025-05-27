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

OBJECT_RADIUS = 20
INITIAL_HP = 20
INITIAL_ATTACK = 5 # From main.py example, ensure this matches your intent
MAX_LAUNCH_STRENGTH = 200 
FORCE_MULTIPLIER = 0.08 
FRICTION = 0.98 
MIN_SPEED_THRESHOLD = 0.1 
ELASTICITY = 0.8 

MAX_VICTORY_POINTS = 5

HP_BAR_WIDTH = 40
HP_BAR_HEIGHT = 5

# --- GameObject Class (Copied from main.py and adapted) ---
class GameObject:
    def __init__(self, x, y, radius, color, hp, attack, player_id, object_id, mass=5.0):
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
        
        # Attributes from main.py's GameObject for drawing/UI, might not be strictly needed for RL state
        self.angle = 0.0
        self.line_length = 0.0
        self.line_thickness = 2
        self.line_color = GREY

    def apply_force(self, dx, dy, strength_multiplier=FORCE_MULTIPLIER):
        # In main.py, dx, dy are from object to mouse (vector for launch)
        # The RL env's step() method calculates dx, dy based on angle and strength.
        # This method should directly use them to set velocity.
        self.vx = -dx * strength_multiplier # Negative because original dx,dy is like mouse_pos - obj_pos
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
            self.vx *= -ELASTICITY
        elif self.x + self.radius > screen_width:
            self.x = screen_width - self.radius
            self.vx *= -ELASTICITY

        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy *= -ELASTICITY
        elif self.y + self.radius > screen_height:
            self.y = screen_height - self.radius
            self.vy *= -ELASTICITY
            # Dampen if hitting bottom slowly (from main.py)
            if abs(self.vy) < MIN_SPEED_THRESHOLD * 2 and self.y + self.radius >= screen_height -1:
                 self.vy = 0
    
    def respawn(self):
        self.hp = self.initial_hp
        self.x = self.original_x
        self.y = self.original_y
        self.vx = 0.0
        self.vy = 0.0
        self.is_moving = False
        self.has_moved_this_turn = False

    def draw(self, surface, font_small): # Added font_small for ID
        if self.hp <= 0:
            return

        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), self.radius, 1)

        if self.hp > 0:
            hp_ratio = self.hp / self.initial_hp
            hp_bar_current_width = int(HP_BAR_WIDTH * hp_ratio)
            hp_bar_color = GREEN if hp_ratio > 0.5 else RED if hp_ratio <= 0.25 else (255,165,0) # Orange

            hp_bar_x = self.x - HP_BAR_WIDTH / 2
            hp_bar_y = self.y - self.radius - HP_BAR_HEIGHT - 5

            pygame.draw.rect(surface, GREY, (hp_bar_x, hp_bar_y, HP_BAR_WIDTH, HP_BAR_HEIGHT))
            pygame.draw.rect(surface, hp_bar_color, (hp_bar_x, hp_bar_y, hp_bar_current_width, HP_BAR_HEIGHT))
        
        id_text_surface = font_small.render(f"{self.object_id+1}", True, BLACK)
        text_rect = id_text_surface.get_rect(center=(self.x, self.y))
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
            self.clock = pygame.time.Clock()
        else: # Headless or for rgb_array
            if not pygame.font.get_init(): pygame.font.init()
            self.font = pygame.font.SysFont(None, 30) # Font might be needed for rgb_array text
            self.font_small = pygame.font.SysFont(None, 18)
            self.screen = None 
            self.clock = None

        self.players_objects: list[list[GameObject]] = []
        self.scores = [0, 0]
        self.current_player_turn = 0
        self.game_over = False
        self.winner = None
        self.action_processing_pending = False # True after a launch, until objects stop
        self.frame_count = 0

        # Attributes from main.py's Game that are UI-specific and not needed for RL game logic:
        # self.selected_object = None
        # self.is_dragging = False
        # self.drag_start_pos = None
        # self.drag_current_pos = None
        
        self._initialize_objects() # Initialize objects upon creation
        # self.reset_game_state() # reset_game_state will also call _initialize_objects

    def _initialize_objects(self):
        # Using spacing from main.py
        spacing = OBJECT_RADIUS * 2 + 20 
        self.players_objects = [
            [GameObject(100 + i * spacing, SCREEN_HEIGHT // 2, OBJECT_RADIUS, BLUE, INITIAL_HP, INITIAL_ATTACK, 0, i) for i in range(self.num_objects_per_player)],
            [GameObject(SCREEN_WIDTH - 100 - i * spacing, SCREEN_HEIGHT // 2, OBJECT_RADIUS, RED, INITIAL_HP, INITIAL_ATTACK, 1, i) for i in range(self.num_objects_per_player)]
        ]

    def reset_game_state(self):
        self.scores = [0, 0]
        self.current_player_turn = random.choice([0,1]) # Random start player
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
        This is derived from main.py's Game.update() method's core logic.
        Returns True if any object was moving or a collision occurred.
        """
        if self.game_over:
            return False

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

                    # Overlap Resolution
                    overlap = min_dist - distance
                    nx = (dist_x / distance) if distance != 0 else 1.0
                    ny = (dist_y / distance) if distance != 0 else 0.0
                    
                    inv_m1 = 1.0 / obj1.mass if obj1.mass > 0 else 0.0
                    inv_m2 = 1.0 / obj2.mass if obj2.mass > 0 else 0.0
                    total_inv_mass = inv_m1 + inv_m2

                    if total_inv_mass > 0:
                        correction_scalar = overlap / total_inv_mass
                        obj1.x += nx * correction_scalar * inv_m1
                        obj1.y += ny * correction_scalar * inv_m1
                        obj2.x -= nx * correction_scalar * inv_m2
                        obj2.y -= ny * correction_scalar * inv_m2

                    # Impulse Calculation
                    rvx = obj1.vx - obj2.vx
                    rvy = obj1.vy - obj2.vy
                    vel_along_normal = rvx * nx + rvy * ny

                    if vel_along_normal < 0: # Moving towards each other
                        e = ELASTICITY
                        impulse_scalar = -(1 + e) * vel_along_normal / total_inv_mass if total_inv_mass > 0 else 0
                        
                        obj1.vx += impulse_scalar * inv_m1 * nx
                        obj1.vy += impulse_scalar * inv_m1 * ny
                        obj2.vx -= impulse_scalar * inv_m2 * nx
                        obj2.vy -= impulse_scalar * inv_m2 * ny
                        
                        obj1.is_moving = True
                        obj2.is_moving = True

                        # Damage Application
                        if obj1.player_id != obj2.player_id: # No friendly fire
                            obj1_hp_before = obj1.hp
                            obj2_hp_before = obj2.hp
                            obj1.hp -= obj2.attack
                            obj2.hp -= obj1.attack

                            if obj1.hp <= 0 and obj1_hp_before > 0:
                                self.scores[obj2.player_id] += 1
                            if obj2.hp <= 0 and obj2_hp_before > 0:
                                self.scores[obj1.player_id] += 1
                    
                    obj1.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT) # Re-check after collision
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
                    obj.respawn()

        player_who_just_moved = self.current_player_turn
        potential_next_player = 1 - player_who_just_moved

        can_current_player_move_again = self.can_player_move(player_who_just_moved)
        can_potential_next_player_move = self.can_player_move(potential_next_player)

        if not can_current_player_move_again: # Current player has no more moves for their launched unit(s)
            if can_potential_next_player_move:
                self.current_player_turn = potential_next_player
            else: # Neither player can make more moves in this "exchange"
                # Reset has_moved_this_turn for all objects for a new exchange
                for p_list_reset in self.players_objects:
                    for o_reset in p_list_reset:
                        o_reset.has_moved_this_turn = False
                # Let the other player start the new exchange
                self.current_player_turn = potential_next_player 
        # If current player *can* move again and opponent cannot, turn stays (implicitly handled)


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
        self.current_player_turn = random.choice([0,1])


    def draw(self): # Adapted from main.py's Game.draw
        if self.headless_mode or not self.screen or not self.font:
            return

        self.screen.fill(LIGHT_GREY)

        for player_objs in self.players_objects:
            for obj in player_objs:
                obj.draw(self.screen, self.font_small) # Pass small font for ID

        # Scores
        score_text_p0 = self.font.render(f"P0 (Blue): {self.scores[0]}", True, BLUE)
        score_text_p1 = self.font.render(f"P1 (Red): {self.scores[1]}", True, RED)
        self.screen.blit(score_text_p0, (10, 10))
        self.screen.blit(score_text_p1, (SCREEN_WIDTH - score_text_p1.get_width() - 10, 10))

        # Turn / Status
        status_text_str = ""
        if not self.game_over:
            player_name = "P0 (Blue)" if self.current_player_turn == 0 else "P1 (Red)"
            if self.action_processing_pending:
                status_text_str = "Objects moving..."
            elif self.can_player_move(self.current_player_turn):
                status_text_str = f"{player_name}'s turn."
            else:
                status_text_str = f"{player_name} has no more moves."
        
        status_color = BLUE if self.current_player_turn == 0 else RED
        status_surf = self.font.render(status_text_str, True, status_color)
        status_rect = status_surf.get_rect(center=(SCREEN_WIDTH // 2, 20))
        self.screen.blit(status_surf, status_rect)

        # Unit status (optional for env, good for human mode)
        y_offset_for_status = 50; x_offset_p1 = 10; x_offset_p2_base = SCREEN_WIDTH - 150
        if self.font_small:
            p0_label = self.font_small.render(f"P0 Units:", True, BLACK)
            self.screen.blit(p0_label, (x_offset_p1, y_offset_for_status))
            current_y_p0 = y_offset_for_status + 15
            for obj in self.players_objects[0]:
                s = "Moved" if obj.has_moved_this_turn else "Ready"
                if obj.hp <=0: s = "KO"
                t = self.font_small.render(f" O{obj.object_id+1}:HP{obj.hp:.0f}({s})", True, BLACK if obj.hp>0 else GREY)
                self.screen.blit(t, (x_offset_p1, current_y_p0)); current_y_p0+=15

            p1_label = self.font_small.render(f"P1 Units:", True, BLACK)
            self.screen.blit(p1_label, (x_offset_p2_base, y_offset_for_status))
            current_y_p1 = y_offset_for_status + 15
            for obj in self.players_objects[1]:
                s = "Moved" if obj.has_moved_this_turn else "Ready"
                if obj.hp <=0: s = "KO"
                t = self.font_small.render(f" O{obj.object_id+1}:HP{obj.hp:.0f}({s})", True, BLACK if obj.hp>0 else GREY)
                self.screen.blit(t, (x_offset_p2_base, current_y_p1)); current_y_p1+=15


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
            candidate_obj = player_objects[obj_choice_idx]
            if candidate_obj.hp > 0 and not candidate_obj.has_moved_this_turn:
                selected_obj = candidate_obj
        
        self.current_reward = 0.0

        if selected_obj:
            angle_rad = (direction_idx / NUM_DIRECTIONS) * 2 * math.pi # 0 to 2PI
            
            # Strengths: 0=low, 1=medium, 2=high. Map to a launch vector magnitude.
            # MAX_LAUNCH_STRENGTH from main.py is the max drag distance.
            # The apply_force takes dx, dy which are like (mouse_release - obj_pos)
            # So, the magnitude of this (dx,dy) vector is what strength controls.
            strength_ratios = [0.33, 0.66, 1.0] # Example: 33%, 66%, 100% of MAX_LAUNCH_STRENGTH
            launch_magnitude = MAX_LAUNCH_STRENGTH * strength_ratios[strength_idx]

            # dx, dy for apply_force are -(mouse_pos - obj_pos) effectively.
            # If we want to launch TOWARDS angle_rad, then the vector from obj to target is (cos, sin)
            # So dx for apply_force should be cos(angle_rad) * launch_magnitude
            # and dy for apply_force should be sin(angle_rad) * launch_magnitude
            # The apply_force method has 'self.vx = -dx * strength_multiplier'.
            # To make vx positive along angle_rad:
            # vx_desired = cos(angle_rad) * effective_velocity_component
            # vy_desired = sin(angle_rad) * effective_velocity_component
            # If apply_force uses vx = -dx * mult, then dx = -vx_desired / mult
            # This is tricky. Let's assume apply_force takes the vector (target_x - obj_x, target_y - obj_y)
            
            # Let dx_action, dy_action be the components of the vector from object towards target
            dx_action = math.cos(angle_rad) * launch_magnitude
            dy_action = math.sin(angle_rad) * launch_magnitude
            
            selected_obj.apply_force(dx_action, dy_action) # GameObject.apply_force uses its own FORCE_MULTIPLIER
            
            self.game.action_processing_pending = True
            self.current_reward -= 0.01 # Small penalty for taking an action
        else:
            self.current_reward -= 0.1 # Penalty for invalid action (e.g., choosing KO'd/moved unit)
            self.game.action_processing_pending = True # Still need to run simulation for turn progression if no valid action

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
                # This logic might be better inside Game.draw if it's aware of headless for rgb_array.
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
