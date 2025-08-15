"""
Common helpers for Seal Slammers environments:
- Lightweight snapshot/restore of env + game numeric state (no pygame surfaces)
- Safe one-step action evaluation via step() with full state restore

Design goals:
- Do not modify the main game/environment classes
- Avoid deep copies of pygame resources
- Be reusable across multiple RL wrappers/policies
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
import copy


def _snapshot_game_object(obj: Any) -> Dict[str, Any]:
    """Capture numeric/stateful attributes needed to fully restore gameplay.

    We intentionally avoid copying any pygame surface/handles.
    """
    return {
        "x": float(obj.x),
        "y": float(obj.y),
        "vx": float(obj.vx),
        "vy": float(obj.vy),
        "is_moving": bool(obj.is_moving),
        "has_moved_this_turn": bool(obj.has_moved_this_turn),
        "hp": float(obj.hp),
        "attack": float(obj.attack),
        "angle": float(getattr(obj, "angle", 0.0)),
        "angular_velocity": float(getattr(obj, "angular_velocity", 0.0)),
        "last_damaged_frame": int(getattr(obj, "last_damaged_frame", 0)),
        # Static fields kept for completeness (they normally don't change):
        "radius": int(getattr(obj, "radius", 23)),
        "player_id": int(getattr(obj, "player_id", 0)),
        "object_id": int(getattr(obj, "object_id", 0)),
        "mass": float(getattr(obj, "mass", 1.0)),
        "restitution": float(getattr(obj, "restitution", 1.0)),
    }


def _restore_game_object(obj: Any, data: Dict[str, Any]) -> None:
    obj.x = float(data["x"])
    obj.y = float(data["y"])
    obj.vx = float(data["vx"])
    obj.vy = float(data["vy"])
    obj.is_moving = bool(data["is_moving"])
    obj.has_moved_this_turn = bool(data["has_moved_this_turn"])
    obj.hp = float(data["hp"])
    obj.attack = float(data["attack"])
    obj.angle = float(data.get("angle", 0.0))
    obj.angular_velocity = float(data.get("angular_velocity", 0.0))
    obj.last_damaged_frame = int(data.get("last_damaged_frame", 0))
    # The below are typically constant per object; set only if present to stay safe
    if "mass" in data:
        obj.mass = float(data["mass"])
    if "restitution" in data:
        obj.restitution = float(data["restitution"])


def snapshot_env_state(env: Any) -> Dict[str, Any]:
    """Snapshot the env and embedded game numeric state.

    Assumptions:
    - `env` exposes `game` with `players_objects`, `scores`, `current_player_turn`, etc.
    - No external resources (windows) are duplicated.
    """
    game = env.game
    players_objs_snap = [
        [_snapshot_game_object(o) for o in player_list]
        for player_list in game.players_objects
    ]

    state = {
        "game": {
            "players_objects": players_objs_snap,
            "scores": [int(game.scores[0]), int(game.scores[1])],
            "current_player_turn": int(game.current_player_turn),
            "game_over": bool(game.game_over),
            "winner": None if game.winner is None else int(game.winner),
            "action_processing_pending": bool(getattr(game, "action_processing_pending", False)),
            "frame_count": int(getattr(game, "frame_count", 0)),
            "first_player_of_round": int(getattr(game, "first_player_of_round", 0)),
            "enemy_collision_happened_this_step": bool(getattr(game, "enemy_collision_happened_this_step", False)),
        },
        "env": {
            "consecutive_meaningless_actions": int(getattr(env, "consecutive_meaningless_actions", 0)),
            "last_opponent_action": copy.deepcopy(getattr(env, "last_opponent_action", [-1.0, -1.0, -1.0])),
            "_hp_snapshot_since_last_turn": copy.deepcopy(getattr(env, "_hp_snapshot_since_last_turn", {0: [], 1: []})),
            "_episode_components": copy.deepcopy(getattr(env, "_episode_components", {})),
            "shaping_scale": float(getattr(env, "shaping_scale", 1.0)),
            "render_mode": getattr(env, "render_mode", None),
        },
    }
    return state


def restore_env_state(env: Any, state: Dict[str, Any]) -> None:
    """Restore env/game numeric state from a snapshot previously produced by snapshot_env_state()."""
    game_state = state["game"]
    env_state = state["env"]

    game = env.game
    # Restore per-object state (assume same structure/order)
    for objs, snap_list in zip(game.players_objects, game_state["players_objects"]):
        for o, o_snap in zip(objs, snap_list):
            _restore_game_object(o, o_snap)

    # Restore game scalars
    game.scores[0] = int(game_state["scores"][0])
    game.scores[1] = int(game_state["scores"][1])
    game.current_player_turn = int(game_state["current_player_turn"])
    game.game_over = bool(game_state["game_over"])
    game.winner = None if game_state["winner"] is None else int(game_state["winner"])
    game.action_processing_pending = bool(game_state["action_processing_pending"])
    game.frame_count = int(game_state["frame_count"])
    game.first_player_of_round = int(game_state["first_player_of_round"])
    game.enemy_collision_happened_this_step = bool(game_state["enemy_collision_happened_this_step"])

    # Restore env fields
    env.consecutive_meaningless_actions = int(env_state["consecutive_meaningless_actions"])
    env.last_opponent_action = copy.deepcopy(env_state["last_opponent_action"])  # type: ignore
    env._hp_snapshot_since_last_turn = copy.deepcopy(env_state["_hp_snapshot_since_last_turn"])  # type: ignore
    env._episode_components = copy.deepcopy(env_state["_episode_components"])  # type: ignore
    env.shaping_scale = float(env_state.get("shaping_scale", getattr(env, "shaping_scale", 1.0)))
    env.render_mode = env_state.get("render_mode", getattr(env, "render_mode", None))


def evaluate_action_reward(env: Any, action: Tuple[int, int, int], step_fn=None) -> float:
    """Evaluate a single action by running a step and restoring state.

    - Temporarily disables rendering to keep evaluation fast
    - Returns the immediate reward from this action (env's own shaping/terminal included)
    - The env is left exactly as it was before the call

    step_fn: optional callable(action) -> (obs, reward, terminated, truncated, info)
             If provided, used instead of env.step to avoid wrapper recursion.
    """
    snap = snapshot_env_state(env)
    prev_render_mode = getattr(env, 'render_mode', None)
    try:
        env.render_mode = None
        if step_fn is None:
            _, reward, _, _, _ = env.step(action)
        else:
            _, reward, _, _, _ = step_fn(action)
        return float(reward)
    except Exception:
        return float('-inf')
    finally:
        try:
            env.render_mode = prev_render_mode
        except Exception:
            pass
        restore_env_state(env, snap)
