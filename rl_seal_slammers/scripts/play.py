import pygame
import numpy as np
import argparse
import sys
import math
import os

# Ensure both package root and project root are on sys.path
_current_dir = os.path.abspath(os.path.dirname(__file__))
_pkg_dir = os.path.abspath(os.path.join(_current_dir, '..'))            # rl_seal_slammers/
_root_dir = os.path.abspath(os.path.join(_current_dir, '..', '..'))     # project root
for _p in (_pkg_dir, _root_dir):
    if _p not in sys.path:
        sys.path.append(_p)

# Fallback-friendly import for shared physics util
try:
    from rl_seal_slammers.physics_utils import simulate_projected_path  # When running from project root
except ImportError:
    try:
        from physics_utils import simulate_projected_path  # When running from inside package dir
    except ImportError:
        simulate_projected_path = None

# Import environment class and constants with fallbacks
try:
    from rl_seal_slammers.envs.seal_slammers_env import (
        SealSlammersEnv,
        SCREEN_WIDTH, SCREEN_HEIGHT,
        OBJECT_RADIUS, MAX_LAUNCH_STRENGTH_RL, MAX_PULL_RADIUS_MULTIPLIER,
        FORCE_MULTIPLIER, FRICTION,
        RED, BLACK, GREEN, BLUE, GREY, LIGHT_GREY,
        ELASTICITY, OBJECT_DEFAULT_HP, OBJECT_DEFAULT_ATTACK
    )
except ImportError:
    try:
        from envs.seal_slammers_env import (
            SealSlammersEnv,
            SCREEN_WIDTH, SCREEN_HEIGHT,
            OBJECT_RADIUS, MAX_LAUNCH_STRENGTH_RL, MAX_PULL_RADIUS_MULTIPLIER,
            FORCE_MULTIPLIER, FRICTION,
            RED, BLACK, GREEN, BLUE, GREY, LIGHT_GREY,
            ELASTICITY, OBJECT_DEFAULT_HP, OBJECT_DEFAULT_ATTACK
        )
    except ImportError as _env_import_err:
        print(f"Failed to import environment and constants: {_env_import_err}")
        sys.exit(1)

# RL algorithms
try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
except ImportError:
    MaskablePPO = None
    PPO = None

# --- Helper Functions ---

def get_human_action(env, player_id):
    """Gets action from a human player using Pygame mouse input."""
    print(f"Player {player_id + 1}'s turn. Select an object to launch.")
    
    selected_object_game_instance = None
    object_game_idx = -1 # Index within the player's list of objects
    
    # Find the game objects for the current player
    player_objects = env.game.players_objects[player_id]

    # Highlight selectable objects for the current player
    original_colors = {}
    for i, obj in enumerate(player_objects):
        if obj.hp > 0 and not env.game.object_has_moved_this_turn(player_id, i): # Check if object can move
            original_colors[i] = obj.color
            obj.color = (255, 255, 0) # Highlight yellow
    env.render() # Re-render to show highlights

    # Object selection phase
    waiting_for_selection = True
    while waiting_for_selection:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for i, obj in enumerate(player_objects):
                    if obj.hp > 0 and not env.game.object_has_moved_this_turn(player_id, i): # Check if object can move
                        distance = math.hypot(mouse_x - obj.x, mouse_y - obj.y)
                        if distance < obj.radius:
                            selected_object_game_instance = obj
                            object_game_idx = i
                            waiting_for_selection = False
                            break
                if selected_object_game_instance:
                    print(f"Selected object {object_game_idx + 1} at ({selected_object_game_instance.x:.0f}, {selected_object_game_instance.y:.0f})")
                    break # Break from event loop once object is selected
        if not waiting_for_selection:
            break
        # env.render() # Keep rendering while waiting - already done by highlight loop or outer loop
        pygame.time.wait(10) # Small delay to prevent busy-waiting

    # Restore original colors
    for i, obj in enumerate(player_objects):
        if i in original_colors:
            obj.color = original_colors[i]
    # env.render() # Render will happen before drag or if no selection

    if not selected_object_game_instance:
        print("No valid object selected or available, passing turn by selecting first available.")
        # Attempt to find first available object if none was clicked
        for i, obj_in_list in enumerate(player_objects):
            if obj_in_list.hp > 0 and not env.game.object_has_moved_this_turn(player_id, i):
                object_game_idx = i
                selected_object_game_instance = obj_in_list
                print(f"Auto-selected object {object_game_idx + 1}")
                break
        if not selected_object_game_instance: # Still no object, e.g. all moved or KO'd
             print("No movable object for human player. Env should handle this state.")
             # This case should ideally be caught by env.can_player_move before calling get_human_action
             # Return a dummy action; the environment should ideally ignore it if the player can't move.
             # A truly robust system might have the play_game loop check env.game.can_player_move()
             # before attempting to get human input.
             return [0, 0, 0] # Default action (first object, 0 angle, 0 strength)

    # Drag and launch phase
    print("Drag to aim and set power, then release.")
    dragging = True
    pull_dx, pull_dy = 0, 0 # Vector from object to mouse (pull direction)
    
    selected_obj_original_color = selected_object_game_instance.color
    selected_object_game_instance.color = (0, 255, 255) # Highlight selected object cyan

    while dragging:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # MOUSEMOTION is handled by re-rendering the line based on current_mouse_x/y above
            # No specific event handling needed for MOUSEMOTION other than updating visuals

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: # Left mouse button up
                    # Use current_mouse_x, current_mouse_y (from pygame.mouse.get_pos())
                    # for consistency with the visual projection, instead of event.pos.
                    pull_dx = current_mouse_x - selected_object_game_instance.x
                    pull_dy = current_mouse_y - selected_object_game_instance.y
                    dragging = False
                    break 
        if not dragging:
            break

        # --- Drawing ---
        # 1. Draw the base game state (clears screen, draws objects, UI)
        current_mouse_x, current_mouse_y = pygame.mouse.get_pos()

        # Update selected object's angle to face the launch direction (opposite of pull)
        obj_center_x_for_angle = selected_object_game_instance.x
        obj_center_y_for_angle = selected_object_game_instance.y
        launch_dir_x_for_angle = obj_center_x_for_angle - current_mouse_x
        launch_dir_y_for_angle = obj_center_y_for_angle - current_mouse_y
        if not (launch_dir_x_for_angle == 0 and launch_dir_y_for_angle == 0):
            selected_object_game_instance.angle = math.atan2(launch_dir_y_for_angle, launch_dir_x_for_angle)

        if env.window_surface: # Should be valid in human render mode
            env.game.draw_game_state(env.window_surface)
        
        # current_mouse_x, current_mouse_y = pygame.mouse.get_pos() # Already got this above

        # --- Drag line and projection logic (adapted & upgraded) ---
        obj_center_x = selected_object_game_instance.x
        obj_center_y = selected_object_game_instance.y
        drag_visual_dx = current_mouse_x - obj_center_x
        drag_visual_dy = current_mouse_y - obj_center_y
        current_pull_distance = math.hypot(drag_visual_dx, drag_visual_dy)
        max_effective_pull_pixels = OBJECT_RADIUS * MAX_PULL_RADIUS_MULTIPLIER
        display_dx = drag_visual_dx
        display_dy = drag_visual_dy
        if current_pull_distance > max_effective_pull_pixels and current_pull_distance > 0:
            ratio = max_effective_pull_pixels / current_pull_distance
            display_dx *= ratio
            display_dy *= ratio
        slingshot_band_end_x = obj_center_x + display_dx
        slingshot_band_end_y = obj_center_y + display_dy
        pygame.draw.line(env.window_surface, RED,(obj_center_x, obj_center_y),(slingshot_band_end_x, slingshot_band_end_y), 3)
        pygame.draw.circle(env.window_surface, RED,(int(slingshot_band_end_x), int(slingshot_band_end_y)), 6)
        # Raw (continuous) pull vector (for angle base)
        raw_pull_dx = drag_visual_dx
        raw_pull_dy = drag_visual_dy
        # 1) Continuous pull angle
        if raw_pull_dx == 0 and raw_pull_dy == 0:
            pull_angle_rad = 0.0
        else:
            pull_angle_rad = math.atan2(raw_pull_dy, raw_pull_dx)
            if pull_angle_rad < 0:
                pull_angle_rad += 2 * math.pi
        # 2) Discretize angle to 72 bins (env uses this pull angle then +pi for launch)
        angle_bin_size = 2 * math.pi / 72
        angle_idx_quant = int(round(pull_angle_rad / angle_bin_size)) % 72
        pull_angle_quant = angle_idx_quant * angle_bin_size
        # 3) Capped pull magnitude in pixels
        capped_pull_pixels = min(current_pull_distance, max_effective_pull_pixels)
        # 4) Discretize strength to 5 levels (0..4) same as action encoding in get_human_action finalize logic
        strength_ratio_cont = 0.0 if max_effective_pull_pixels == 0 else capped_pull_pixels / max_effective_pull_pixels
        strength_idx_quant = int(round(strength_ratio_cont * 4))
        if strength_idx_quant < 0: strength_idx_quant = 0
        if strength_idx_quant > 4: strength_idx_quant = 4
        # 5) Map strength_idx -> strength_scale identical to env.step: (idx+1)/5.0
        strength_scale_quant = (strength_idx_quant + 1) / 5.0
        # 6) Convert to actual launch magnitude (velocity scalar)
        actual_strength_magnitude_quant = strength_scale_quant * MAX_LAUNCH_STRENGTH_RL
        # 7) Launch angle is opposite to pull direction
        launch_angle_quant = pull_angle_quant + math.pi
        # 8) Quantized initial velocity (what env will reconstruct)
        init_vx_quant = math.cos(launch_angle_quant) * actual_strength_magnitude_quant
        init_vy_quant = math.sin(launch_angle_quant) * actual_strength_magnitude_quant
        # 9) Draw discrete (quantized) predicted trajectory so it matches env.step
        if strength_idx_quant >= 0 and (abs(init_vx_quant) > 1e-6 or abs(init_vy_quant) > 1e-6):
            if simulate_projected_path:
                projected_points = simulate_projected_path(
                    selected_object_game_instance,
                    env.game.players_objects,
                    init_vx_quant,
                    init_vy_quant,
                    steps=120,
                    friction=FRICTION,
                    min_speed=0.1,
                    screen_width=SCREEN_WIDTH,
                    screen_height=SCREEN_HEIGHT,
                )
                for i, (px, py) in enumerate(projected_points):
                    if i % 3 == 0:
                        pygame.draw.circle(env.window_surface, BLACK, (int(px), int(py)), 2)
        # --- End quantized projection ---

        pygame.display.flip()

        # --- Clock Tick ---
        if env.clock:
            env.clock.tick(env.metadata['render_fps'])
        else:
            # Fallback, though env.clock should be initialized by env.render()
            # if render_mode is human and it has been called at least once.
            pygame.time.Clock().tick(60) # Default to 60 FPS


    selected_object_game_instance.color = selected_obj_original_color # Restore color
    env.render() # Render final state before action is processed by env

    # Convert pull_dx, pull_dy to angle and strength for the environment's action space
    
    pull_angle_rad = 0.0
    strength_abs = 0.0

    # Calculate pull_dx, pull_dy based on the final mouse position relative to object center
    # This should use the *actual* pull vector, capped by max_effective_pull_pixels for strength calculation
    final_pull_dx = pull_dx # pull_dx was from MOUSEBUTTONUP event: final_mouse_x - selected_object_game_instance.x
    final_pull_dy = pull_dy # pull_dy was from MOUSEBUTTONUP event: final_mouse_y - selected_object_game_instance.y
    
    final_pull_magnitude = math.hypot(final_pull_dx, final_pull_dy)

    if final_pull_magnitude == 0:
        pull_angle_rad = 0.0
        strength_abs = 0.0
    else:
        pull_angle_rad = math.atan2(final_pull_dy, final_pull_dx)
        # Cap the strength_abs by max_effective_pull_pixels for action calculation
        strength_abs = min(final_pull_magnitude, max_effective_pull_pixels)
        # If the capped pull is very small, treat as no launch (consistent with projection logic)
        if strength_abs < 5: # Threshold from main.py
            strength_abs = 0.0

    # Normalize pull_angle_rad to 0-2pi
    if pull_angle_rad < 0:
        pull_angle_rad += 2 * math.pi

    # Discretize pull_angle_rad (72 directions for action space)
    # This angle_idx will be interpreted by the env as the pull angle.
    angle_idx = round(pull_angle_rad / (2 * math.pi / 72)) % 72

    # Discretize strength (5 levels for action space)
    visual_max_pull_strength = OBJECT_RADIUS * MAX_PULL_RADIUS_MULTIPLIER 
    if visual_max_pull_strength == 0:
        strength_ratio = 0.0
    else:
        strength_ratio = min(strength_abs / visual_max_pull_strength, 1.0)
    strength_idx = int(round(strength_ratio * 4)) # 0..4
    
    action = [object_game_idx, angle_idx, strength_idx]
    print(f"Player {player_id + 1} action: Object {action[0]+1}, Pull Angle Index {action[1]}/72, Strength Index {action[2]}")
    return action


def get_random_ai_action(env, player_id): # Renamed from get_ai_action
    """Gets action from a simple AI (random valid action)."""
    # Find a random alive object for the current player
    player_objects = env.game.players_objects[player_id]
    
    # Get indices of objects that are alive AND haven\'t moved yet
    available_object_indices = [
        i for i, obj in enumerate(player_objects) 
        if obj.hp > 0 and not env.game.object_has_moved_this_turn(player_id, i)
    ]

    if not available_object_indices:
        # This case should ideally be prevented by checking env.game.can_player_move() in the main loop
        # before calling get_ai_action. If called, it means no valid move can be made.
        # Return a default action (e.g., first object, zero angle/strength) or handle as appropriate.
        # The environment step function should also gracefully handle actions on already moved/KO'd objects.
        print(f"AI Player {player_id + 1} has no valid moves. Returning default action.")
        return [0, 0, 0] # Default action: obj 0, angle_idx 0, strength_idx 0

    selected_object_idx_in_player_list = np.random.choice(available_object_indices)
    
    # Random angle and strength
    angle_idx = np.random.randint(0, env.action_space.nvec[1]) # 0-71 for angle (env.action_space.nvec[1] will be 72)
    strength_idx = np.random.randint(0, env.action_space.nvec[2]) # 0-4 for strength
    
    action = [selected_object_idx_in_player_list, angle_idx, strength_idx]
    print(f"AI Player {player_id + 1} (Random) action: Object {action[0]+1}, Pull Angle Index {action[1]}/72, Strength Index {action[2]}")
    return action

# --- Main Game Loop ---

def play_game(mode, num_objects=3, model_path=None,
              p0_hp=None, p0_atk=None, p1_hp=None, p1_atk=None): # Added HP/ATK params
    print(f"[DEBUG play_game] Initializing game with mode: {mode}, num_objects: {num_objects}, model_path: {model_path}, p0_hp: {p0_hp}, p0_atk: {p0_atk}, p1_hp: {p1_hp}, p1_atk: {p1_atk}") # DEBUG
    render_mode = 'human' if mode != 'ai_vs_ai_fast' else 'rgb_array'
    
    env_kwargs = {
        'num_objects_per_player': num_objects,
        # Pass along HP/ATK. SealSlammersEnv will use these if provided,
        # otherwise, it will randomize for training, or use its internal defaults if these are None
        # and randomization is not its primary mode (which it is for training).
        # For play.py, we want to use these fixed values if provided by CLI.
        'p0_hp_fixed': p0_hp, 
        'p0_atk_fixed': p0_atk,
        'p1_hp_fixed': p1_hp,
        'p1_atk_fixed': p1_atk
    }

    if mode == 'ai_vs_ai_fast':
        print("Running AI vs AI (fast mode - minimal rendering if any through env)")
        env = SealSlammersEnv(render_mode=None, **env_kwargs)
    else:
        env = SealSlammersEnv(render_mode='human', **env_kwargs)

    # Load PPO model if path is provided and AI is involved
    ppo_model = None
    if model_path and ('ai' in mode):
        if os.path.exists(model_path):
            print(f"Loading PPO model from: {model_path}")
            try:
                # Try MaskablePPO first (matches training script)
                ppo_model = MaskablePPO.load(model_path)
                print("Model loaded with MaskablePPO.")
            except Exception as e_mask:
                print(f"MaskablePPO load failed: {e_mask}. Trying stable-baselines3 PPO...")
                try:
                    ppo_model = PPO.load(model_path)
                    print("Model loaded with PPO.")
                except Exception as e_ppo:
                    print(f"Error loading PPO model: {e_ppo}. Falling back to random AI.")
                    ppo_model = None
        else:
            print(f"Model path not found: {model_path}. Falling back to random AI.")

    obs, info = env.reset()
    print(f"[DEBUG play_game] env.reset() called. Initial obs shape: {obs.shape if isinstance(obs, np.ndarray) else type(obs)}, info: {info}") # DEBUG
    print(f"[DEBUG play_game] Initial game.current_player_turn from env after reset: {env.game.current_player_turn}") # DEBUG
    terminated = False
    truncated = False
    print(f"Initial observation received. Game reset. Current player: {env.game.current_player_turn + 1}")

    player_types = []
    if mode == 'human_vs_human':
        player_types = ['human', 'human']
    elif mode == 'human_vs_ai':
        player_types = ['human', 'ai'] # Human is P1 (Blue)
    elif mode == 'ai_vs_human':
        player_types = ['ai', 'human'] # Human is P2 (Red)
    elif mode == 'ai_vs_ai' or mode == 'ai_vs_ai_fast':
        player_types = ['ai', 'ai']
    else:
        raise ValueError(f"Unknown game mode: {mode}")
    print(f"[DEBUG play_game] Player types set to: P0='{player_types[0]}', P1='{player_types[1]}'") # DEBUG

    running = True
    while running:
        print(f"\\n[DEBUG play_game] START OF TURN. env.game.current_player_turn: {env.game.current_player_turn}") # DEBUG
        if env.render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break
        
        current_player_id = env.game.current_player_turn
        player_type = player_types[current_player_id]
        print(f"[DEBUG play_game] Determined current_player_id: {current_player_id} ({player_type})") # DEBUG

        action = None
        # Check if the current player can actually move before soliciting action
        if not env.game.can_player_move(current_player_id):
            print(f"Player {current_player_id + 1} ({player_type}) cannot move. Advancing state.")
            action = [0,0,0] # Dummy action
        elif player_type == 'human':
            action = get_human_action(env, current_player_id)
        elif player_type == 'ai':
            if ppo_model:
                # Use PPO model to predict action
                print(f"AI Player {current_player_id + 1} (PPO Model) is thinking...")
                try:
                    # If using MaskablePPO, pass the current action mask so it respects invalid actions
                    if isinstance(ppo_model, MaskablePPO):
                        current_mask = None
                        try:
                            current_mask = env.action_masks()
                        except Exception as e_mask_fetch:
                            print(f"[WARN] Failed to fetch action mask for predict: {e_mask_fetch}")
                            current_mask = None
                        if current_mask is not None:
                            action_array, _ = ppo_model.predict(
                                obs,
                                deterministic=True,
                                action_masks=current_mask,
                            )
                        else:
                            action_array, _ = ppo_model.predict(
                                obs,
                                deterministic=True,
                            )
                    else:
                        # Standard PPO does not take action masks
                        action_array, _ = ppo_model.predict(
                            obs,
                            deterministic=True,
                        )
                except Exception as e_pred:
                    print(f"[ERROR] PPO predict failed: {e_pred}. Falling back to random action.")
                    action_array = np.array(get_random_ai_action(env, current_player_id))
                action = action_array.tolist() 
                # episode_start_ai = False # Removed for MlpPolicy
                print(f"AI Player {current_player_id + 1} (PPO Model) action: Object {action[0]+1}, Pull Angle Index {action[1]}/72, Strength Index {action[2]}")
            else:
                # Fallback to random AI if model not loaded
                action = get_random_ai_action(env, current_player_id)
        
        if action is None: 
            print("Error: No action decided. Using random action.")
            action = env.action_space.sample().tolist()
        
        print(f"[DEBUG play_game] PRE-STEP: Applying action for player {current_player_id+1} ({player_type}). Action: {action}") # DEBUG
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"[DEBUG play_game] POST-STEP: Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}") # DEBUG
      
        if env.render_mode == 'human' or (mode == 'ai_vs_ai_fast' and (terminated or truncated)): # Render last frame for fast mode
            env.render()

        if terminated or truncated:
            print("Game Over!")
            if info.get('winner') is not None:
                print(f"Player {info['winner'] + 1} wins!")
            else:
                print("It's a draw or truncated!")
            print(f"Final Scores: P1: {info['scores'][0]}, P2: {info['scores'][1]}")
            
            # If we were to play again in a loop here, after env.reset(), we would set:
            # lstm_state_ai = None
            # episode_start_ai = True

            # Simple wait loop after game over before closing, or ask to play again
            if env.render_mode == 'human':
                print("Close window or press ESC to exit.")
                wait_for_exit = True
                while wait_for_exit:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            wait_for_exit = False
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            wait_for_exit = False
                    if not wait_for_exit: break
            running = False # End the main game loop

    env.close()
    pygame.quit()
    sys.exit()


def _simulate_projected_path_play(env, source_obj, init_vx, init_vy, steps=120):
    return simulate_projected_path(
        source_obj,
        env.game.players_objects,
        init_vx,
        init_vy,
        steps=steps,
        friction=FRICTION,
        min_speed=0.1,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Seal Slammers.")
    parser.add_argument("--mode", type=str, default="human_vs_human",
                        choices=['human_vs_human', 'human_vs_ai', 'ai_vs_human', 'ai_vs_ai', 'ai_vs_ai_fast'],
                        help="Game mode (default: human_vs_human). 'ai_vs_ai_fast' runs without human-speed rendering.")
    parser.add_argument("--objects", type=int, default=3, help="Number of objects per player (default: 3). Max usually 3.")
    
    # Add arguments for HP and Attack for Player 0 (Human in human_vs_ai)
    parser.add_argument("--p0_hp", type=int, default=40, help=f"HP for Player 0 objects (default: env default/randomized, currently {OBJECT_DEFAULT_HP} if not randomized by env for training).")
    parser.add_argument("--p0_atk", type=int, default=8, help=f"Attack for Player 0 objects (default: env default/randomized, currently {OBJECT_DEFAULT_ATTACK} if not randomized by env for training).")
    
    # Add arguments for HP and Attack for Player 1 (AI in human_vs_ai)
    parser.add_argument("--p1_hp", type=int, default=40, help=f"HP for Player 1 objects (default: env default/randomized, currently {OBJECT_DEFAULT_HP} if not randomized by env for training).")
    parser.add_argument("--p1_atk", type=int, default=8, help=f"Attack for Player 1 objects (default: env default/randomized, currently {OBJECT_DEFAULT_ATTACK} if not randomized by env for training).")

    # Determine project root to construct default model path
    # play.py is in RL_Seal_Slammers/rl_seal_slammers/scripts/play.py
    # Project root is RL_Seal_Slammers/
    project_root_for_play_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Updated default model path to align with the new training script's output (MaskablePPO)
    default_model_path = os.path.join(project_root_for_play_script, "models", "sb3_maskable_ppo_sealslammers_mlp", "maskable_ppo_sealslammers_mlp_model_final.zip")
    parser.add_argument("--model-path", type=str, default=default_model_path,
                        help=f"Path to the trained PPO model .zip file (default: {default_model_path}). Used if mode involves AI.")
    
    args = parser.parse_args()

    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()

    play_game(args.mode, args.objects, args.model_path,
              args.p0_hp, args.p0_atk, args.p1_hp, args.p1_atk) # Pass new args
