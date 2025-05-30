import pygame
import numpy as np
import argparse
import sys
import math

# Add the parent directory to sys.path to allow imports from envs
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.seal_slammers_env import SealSlammersEnv, SCREEN_WIDTH, SCREEN_HEIGHT, OBJECT_RADIUS, MAX_LAUNCH_STRENGTH_RL, MAX_PULL_RADIUS_MULTIPLIER, FORCE_MULTIPLIER, FRICTION, RED, BLACK, GREEN, BLUE, GREY, LIGHT_GREY, ELASTICITY

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

        # --- Drag line and projection logic (adapted from main.py) ---
        obj_center_x = selected_object_game_instance.x
        obj_center_y = selected_object_game_instance.y

        drag_visual_dx = current_mouse_x - obj_center_x
        drag_visual_dy = current_mouse_y - obj_center_y
        current_pull_distance = math.hypot(drag_visual_dx, drag_visual_dy)
        max_effective_pull_pixels = OBJECT_RADIUS * MAX_PULL_RADIUS_MULTIPLIER

        display_dx = drag_visual_dx
        display_dy = drag_visual_dy

        if current_pull_distance > max_effective_pull_pixels:
            ratio = max_effective_pull_pixels / current_pull_distance
            display_dx *= ratio
            display_dy *= ratio
        
        # Draw the slingshot band (from object towards mouse, capped visually)
        slingshot_band_end_x = obj_center_x + display_dx
        slingshot_band_end_y = obj_center_y + display_dy
        pygame.draw.line(env.window_surface, RED, 
                         (obj_center_x, obj_center_y),
                         (slingshot_band_end_x, slingshot_band_end_y), 3)
        pygame.draw.circle(env.window_surface, RED, 
                           (int(slingshot_band_end_x), int(slingshot_band_end_y)), 6)

        # --- Projected path (dotted line, adapted from main.py) ---
        # The launch force is opposite to the pull direction.
        launch_force_dir_x = obj_center_x - current_mouse_x # Inverted from drag_visual_dx for launch
        launch_force_dir_y = obj_center_y - current_mouse_y # Inverted from drag_visual_dy for launch

        # Cap the pull distance for force calculation (used for projection and final action)
        actual_pull_distance_for_force = current_pull_distance
        if actual_pull_distance_for_force > max_effective_pull_pixels:
            actual_pull_distance_for_force = max_effective_pull_pixels
        
        # If pull is too small, consider it zero for projection (matches main.py behavior)
        if actual_pull_distance_for_force < 5: # Threshold from main.py
            scaled_launch_force_dx = 0
            scaled_launch_force_dy = 0
        else:
            # Scale the launch_force_dir vector by the (capped) actual_pull_distance_for_force
            # The launch_force_dir is already pointing in the launch direction.
            # We need its unit vector multiplied by the effective pull strength.
            dir_len = math.hypot(launch_force_dir_x, launch_force_dir_y)
            if dir_len > 0:
                scaled_launch_force_dx = (launch_force_dir_x / dir_len) * actual_pull_distance_for_force
                scaled_launch_force_dy = (launch_force_dir_y / dir_len) * actual_pull_distance_for_force
            else:
                scaled_launch_force_dx = 0
                scaled_launch_force_dy = 0

        # Projection parameters from envs.seal_slammers_env constants
        # FORCE_MULTIPLIER is defined in envs.seal_slammers_env
        # FRICTION is defined in envs.seal_slammers_env
        # We need to import them or pass them if they are not already available here.
        # For now, assuming they are available via env.game or env directly if they were constants there.
        # Let's use the constants directly from the import for now.
        # from envs.seal_slammers_env import FORCE_MULTIPLIER, FRICTION (already imported)

        proj_vx = scaled_launch_force_dx * FORCE_MULTIPLIER 
        proj_vy = scaled_launch_force_dy * FORCE_MULTIPLIER

        temp_x, temp_y = obj_center_x, obj_center_y
        num_projection_points = 60 
        # FRICTION is already available from envs.seal_slammers_env import
        # proj_radius = selected_object_game_instance.radius # Not strictly needed for point projection

        if not (scaled_launch_force_dx == 0 and scaled_launch_force_dy == 0):
            # Use a small radius for projection points for boundary checks, can be 0 if only center is checked
            proj_point_radius = 1 # Effectively a point for collision
            for i in range(num_projection_points):
                temp_x += proj_vx
                temp_y += proj_vy
                proj_vx *= FRICTION 
                proj_vy *= FRICTION

                # Boundary collision for projection
                if temp_x - proj_point_radius < 0:
                    temp_x = proj_point_radius
                    proj_vx *= -ELASTICITY 
                elif temp_x + proj_point_radius > SCREEN_WIDTH:
                    temp_x = SCREEN_WIDTH - proj_point_radius
                    proj_vx *= -ELASTICITY
                
                if temp_y - proj_point_radius < 0:
                    temp_y = proj_point_radius
                    proj_vy *= -ELASTICITY
                elif temp_y + proj_point_radius > SCREEN_HEIGHT:
                    temp_y = SCREEN_HEIGHT - proj_point_radius
                    proj_vy *= -ELASTICITY

                if i % 5 == 0: # Draw a dot every 5 points for a dashed line effect
                    pygame.draw.circle(env.window_surface, BLACK, (int(temp_x), int(temp_y)), 2)
        # --- End of projection ---        

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
    angle_idx = round(pull_angle_rad / (2 * math.pi / 72)) % 72 # Updated to 72 directions

    # Discretize strength (5 levels for action space)
    # Use MAX_PULL_RADIUS_MULTIPLIER for scaling, consistent with main.py's concept
    visual_max_pull_strength = OBJECT_RADIUS * MAX_PULL_RADIUS_MULTIPLIER 
    
    if visual_max_pull_strength == 0: # Avoid division by zero if OBJECT_RADIUS is 0
        strength_ratio = 0.0
    else:
        strength_ratio = min(strength_abs / visual_max_pull_strength, 1.0)
    
    # Map strength_ratio (0.0 to 1.0) to strength_idx (0 to 4)
    # If strength_abs is very small (e.g., less than a pixel, or if it was 0 initially),
    # strength_ratio will be close to 0, and strength_idx will be 0.
    # This corresponds to the minimum launch power in the env (0.2 * MAX_LAUNCH_STRENGTH_RL).
    # This behavior is slightly different from main.py's "no launch if pull < 5 pixels",
    # but aligns with the discrete action space of the RL environment.
    strength_idx = int(round(strength_ratio * 4)) # round() for better mapping: 0.0->0, 0.25->1, 0.5->2, 0.75->3, 1.0->4

    action = [object_game_idx, angle_idx, strength_idx]
    print(f"Player {player_id + 1} action: Object {action[0]+1}, Pull Angle Index {action[1]}/72, Strength Index {action[2]}")
    return action


def get_ai_action(env, player_id):
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
    print(f"AI Player {player_id + 1} action: Object {action[0]+1}, Pull Angle Index {action[1]}/72, Strength Index {action[2]}")
    return action

# --- Main Game Loop ---

def play_game(mode, num_objects=3):
    render_mode = 'human' if mode != 'ai_vs_ai_fast' else 'rgb_array' # Human for most, fast for ai-ai
    if mode == 'ai_vs_ai_fast':
        print("Running AI vs AI (fast mode - minimal rendering if any through env)")
        env = SealSlammersEnv(render_mode=None, num_objects_per_player=num_objects)
    else:
        env = SealSlammersEnv(render_mode='human', num_objects_per_player=num_objects)

    obs, info = env.reset()
    terminated = False
    truncated = False
    
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

    running = True
    while running:
        if env.render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break
        
        current_player_id = env.game.current_player_turn
        player_type = player_types[current_player_id]

        action = None
        # Check if the current player can actually move before soliciting action
        if not env.game.can_player_move(current_player_id):
            print(f"Player {current_player_id + 1} ({player_type}) cannot move. Advancing state.")
            # If a player cannot move, the environment\'s step function should ideally handle this
            # by just advancing the turn/round. We send a dummy action.
            # The environment step() should check if selected_obj is valid and hasn\'t moved.
            # If not, it proceeds to turn/round logic without applying force.
            action = [0,0,0] # Dummy action (first object, 0 angle, 0 strength)
        elif player_type == 'human':
            action = get_human_action(env, current_player_id)
        elif player_type == 'ai':
            action = get_ai_action(env, current_player_id)
        
        if action is None: # Should not happen if logic is correct
            print("Error: No action decided. Skipping turn.") # Or exit
            # Potentially pass a no-op action if env supports it or skip
            # For now, let's assume a random action if this state is reached.
            action = env.action_space.sample()


        obs, reward, terminated, truncated, info = env.step(action)
        
        if env.render_mode == 'human' or (mode == 'ai_vs_ai_fast' and (terminated or truncated)): # Render last frame for fast mode
            env.render()

        if terminated or truncated:
            print("Game Over!")
            if info.get('winner') is not None:
                print(f"Player {info['winner'] + 1} wins!")
            else:
                print("It's a draw or truncated!")
            print(f"Final Scores: P1: {info['scores'][0]}, P2: {info['scores'][1]}")
            
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Seal Slammers.")
    parser.add_argument("--mode", type=str, default="human_vs_human",
                        choices=['human_vs_human', 'human_vs_ai', 'ai_vs_human', 'ai_vs_ai', 'ai_vs_ai_fast'],
                        help="Game mode (default: human_vs_human). 'ai_vs_ai_fast' runs without human-speed rendering.")
    parser.add_argument("--objects", type=int, default=3, help="Number of objects per player (default: 3). Max usually 3.")
    
    args = parser.parse_args()

    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()

    play_game(args.mode, args.objects)
