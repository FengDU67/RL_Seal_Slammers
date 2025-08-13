import math
from typing import List, Sequence, Tuple

# Shared lightweight physics projection for trajectory preview.
# This mirrors the per-frame ordering used in main game & env physics:
# position update -> friction -> boundary collisions -> pairwise collisions (overlap resolve + impulse) -> boundary re-check via caller if needed.

def simulate_projected_path(source_obj, players_objects: Sequence[Sequence], init_vx: float, init_vy: float, 
                            steps: int = 120, *, friction: float = 0.98, min_speed: float = 0.1,
                            screen_width: int = 1000, screen_height: int = 600) -> list[tuple[float, float]]:
    """Simulate and return list of (x,y) for the launching object's predicted path.
    - source_obj: the real object instance selected (must have x,y,radius,player_id,object_id,mass,restitution,hp attributes)
    - players_objects: nested list of all teams' objects (like game.players_objects)
    - init_vx, init_vy: initial launch velocities applied to source object
    - steps: max simulation frames
    - friction: linear damping per frame
    - min_speed: threshold below which velocity is zeroed (component magnitude check effect approximated by speed magnitude)
    The simulation clones minimal object state; it also imparts velocities to other objects when collisions occur so that
    post-collision deflection distance is not overestimated.
    """
    class SimO:
        __slots__ = ("x","y","vx","vy","r","mass","rest","alive","pid","oid")
        def __init__(self, o):
            self.x = float(o.x); self.y = float(o.y)
            self.vx = 0.0; self.vy = 0.0
            self.r = getattr(o, 'radius', 0)
            self.mass = float(getattr(o, 'mass', 6.0))
            self.rest = float(getattr(o, 'restitution', 1.0))
            self.alive = (getattr(o, 'hp', 1) > 0)
            self.pid = getattr(o, 'player_id', 0)
            self.oid = getattr(o, 'object_id', 0)
    # Clone all objects
    sim_objs = [[SimO(o) for o in team] for team in players_objects]

    # Locate cloned source
    sim_source = None
    for team in sim_objs:
        for so in team:
            if so.pid == getattr(source_obj, 'player_id', -1) and so.oid == getattr(source_obj, 'object_id', -1):
                sim_source = so; break
        if sim_source: break
    if sim_source is None:
        return []
    sim_source.vx = init_vx; sim_source.vy = init_vy

    points: list[Tuple[float, float]] = []
    for _ in range(steps):
        any_motion = False
        # Move & friction & boundary
        for team in sim_objs:
            for so in team:
                if not so.alive:
                    continue
                if abs(so.vx) > 1e-6 or abs(so.vy) > 1e-6:
                    so.x += so.vx; so.y += so.vy
                    so.vx *= friction; so.vy *= friction
                    # Stop check (approx speed-based)
                    if abs(so.vx) < min_speed and abs(so.vy) < min_speed:
                        so.vx = 0.0; so.vy = 0.0
                    else:
                        any_motion = True
                # Boundary collisions
                if so.x - so.r < 0:
                    so.x = so.r; so.vx *= -so.rest
                elif so.x + so.r > screen_width:
                    so.x = screen_width - so.r; so.vx *= -so.rest
                if so.y - so.r < 0:
                    so.y = so.r; so.vy *= -so.rest
                elif so.y + so.r > screen_height:
                    so.y = screen_height - so.r; so.vy *= -so.rest
        # Pairwise collisions
        flat = [so for team in sim_objs for so in team if so.alive]
        for i in range(len(flat)):
            a = flat[i]
            for j in range(i+1, len(flat)):
                b = flat[j]
                dx = a.x - b.x; dy = a.y - b.y
                dist = math.hypot(dx, dy)
                if dist == 0:
                    continue
                min_d = a.r + b.r
                if dist < min_d:
                    overlap = min_d - dist
                    nx = dx / dist; ny = dy / dist
                    inv_ma = 1.0 / a.mass if a.mass > 0 else 0.0
                    inv_mb = 1.0 / b.mass if b.mass > 0 else 0.0
                    inv_sum = inv_ma + inv_mb
                    if inv_sum > 0:
                        a.x += nx * overlap * (inv_ma / inv_sum)
                        a.y += ny * overlap * (inv_ma / inv_sum)
                        b.x -= nx * overlap * (inv_mb / inv_sum)
                        b.y -= ny * overlap * (inv_mb / inv_sum)
                    # Relative velocity along normal
                    rvx = a.vx - b.vx; rvy = a.vy - b.vy
                    vel_norm = rvx * nx + rvy * ny
                    if vel_norm < 0:
                        e = a.rest if a.rest < b.rest else b.rest
                        j_imp = -(1 + e) * vel_norm / (inv_sum if inv_sum > 0 else 1.0)
                        if inv_ma > 0:
                            a.vx += j_imp * inv_ma * nx
                            a.vy += j_imp * inv_ma * ny
                        if inv_mb > 0:
                            b.vx -= j_imp * inv_mb * nx
                            b.vy -= j_imp * inv_mb * ny
                        any_motion = True
        points.append((sim_source.x, sim_source.y))
        if not any_motion:
            break
    return points
