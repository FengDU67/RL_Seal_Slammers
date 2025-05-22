import pygame
import random
import math
from typing import List, Optional

# --- 常量 ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# 游戏参数
OBJECT_RADIUS = 20
INITIAL_HP = 100
INITIAL_ATTACK = 10
MAX_VICTORY_POINTS = 5 # 胜利点数

# --- 游戏对象类 ---
class GameObject:
    def __init__(self, x, y, radius, color, hp, attack, player_id, object_id):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.initial_hp = hp
        self.hp = hp
        self.attack = attack
        self.player_id = player_id  # 0 for player 1 (left), 1 for player 2 (right)
        self.object_id = object_id # Unique ID for the object within its team
        self.vx = 0  # X方向速度
        self.vy = 0  # Y方向速度
        self.is_moving = False
        self.has_moved_this_turn = False
        self.original_x = x # For respawning
        self.original_y = y # For respawning
        self.angle = 0.0 # 物体当前角度 (弧度)
        self.angular_velocity = 0.0 # 物体角速度 (弧度/帧)
        self.mass = 5.0 # 质量 (可调)
        self.restitution = 0.8 # 弹性系数 (0-1, 越大越弹, 可调)
        self.moment_of_inertia = 0.5 * self.mass * self.radius**2
        self.last_damaged_frame = -1000 # 上次受到伤害的帧数，初始化为很久以前
        self.damage_intake_cooldown_frames = 15 # 受到伤害后的冷却帧数 (例如 15帧 ~= 0.25秒 @60FPS, 可调)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

        # 显示HP和攻击力
        font = pygame.font.SysFont(None, 18) # 调整字体大小

        hp_text_surface = font.render(f"HP: {self.hp}", True, BLACK)
        atk_text_surface = font.render(f"ATK: {self.attack}", True, BLACK)

        spacing = 5 # HP 和 ATK 文本之间的间距
        total_text_block_width = hp_text_surface.get_width() + spacing + atk_text_surface.get_width()

        # 计算HP文本的起始X坐标，使整个文本块在圆形下方居中
        hp_start_x = self.x - total_text_block_width / 2
        atk_start_x = hp_start_x + hp_text_surface.get_width() + spacing
        text_y_pos = self.y + self.radius + 5 # Y轴位置，圆形下方再加一点间距

        screen.blit(hp_text_surface, (hp_start_x, text_y_pos))
        screen.blit(atk_text_surface, (atk_start_x, text_y_pos))

        # 绘制朝向指示线
        line_end_x = self.x + self.radius * math.cos(self.angle)
        line_end_y = self.y + self.radius * math.sin(self.angle)
        pygame.draw.line(screen, BLACK, (self.x, self.y), (line_end_x, line_end_y), 2)

        if self.has_moved_this_turn: # 如果本回合已行动，则高亮显示
            pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), self.radius + 3, 2) # 增加高亮半径和线宽

    def move(self):
        if self.is_moving:
            self.x += self.vx
            self.y += self.vy

            # 减速 (摩擦力)
            self.vx *= 0.98 # Slightly reduced linear friction
            self.vy *= 0.98 # Slightly reduced linear friction

            # 如果速度很小，则停止移动
            if abs(self.vx) < 0.1 and abs(self.vy) < 0.1:
                self.vx = 0
                self.vy = 0
                self.is_moving = False

        # 更新角度并应用角摩擦力
        self.angle += self.angular_velocity
        self.angular_velocity *= 0.97 # 角速度摩擦/阻尼

    def check_boundary_collision(self, screen_width, screen_height):
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx *= -1
        elif self.x + self.radius > screen_width:
            self.x = screen_width - self.radius
            self.vx *= -1
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy *= -1
        elif self.y + self.radius > screen_height:
            self.y = screen_height - self.radius
            self.vy *= -1

    def apply_force(self, dx, dy, strength_multiplier=0.08): # Slightly increased launch strength
        self.vx = -dx * strength_multiplier
        self.vy = -dy * strength_multiplier
        self.is_moving = True
        self.has_moved_this_turn = True

    def take_damage(self, amount, current_game_frame):
        if current_game_frame - self.last_damaged_frame < self.damage_intake_cooldown_frames:
            return False # 处于冷却中，未造成伤害

        self.hp -= amount
        self.last_damaged_frame = current_game_frame # 更新上次受伤的帧
        if self.hp < 0:
            self.hp = 0
        return True # 成功造成伤害

    def respawn(self):
        self.x = self.original_x
        self.y = self.original_y
        self.hp = self.initial_hp
        self.vx = 0
        self.vy = 0
        self.is_moving = False
        self.angle = 0.0
        self.angular_velocity = 0.0
        # self.mass and self.moment_of_inertia remain as initialized
        # self.has_moved_this_turn will be reset at the start of a player's turn or round


# --- 游戏主类 ---
class Game:
    def __init__(self):
        print("DEBUG: Game.__init__ called") # DEBUG
        pygame.init()
        print("DEBUG: pygame.init() successful") # DEBUG
        try:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            print("DEBUG: pygame.display.set_mode() successful") # DEBUG
        except pygame.error as e:
            print(f"DEBUG: Error calling pygame.display.set_mode: {e}") # DEBUG
            raise
        pygame.display.set_caption("弹射对战游戏")
        self.clock = pygame.time.Clock()
        print("DEBUG: Clock initialized") # DEBUG
        try:
            self.font = pygame.font.SysFont(None, 30)
            print("DEBUG: Font initialized") # DEBUG
        except Exception as e:
            print(f"DEBUG: Error initializing font: {e}") # DEBUG
            # Decide if font is critical; for now, let it proceed or raise
            # raise

        self.players_objects: List[List[GameObject]] = [[], []] # 存储两方玩家的object
        self.scores = [0, 0] # 存储两方玩家的胜利点

        self.setup_objects()

        self.current_player_turn = random.choice([0, 1]) # 0: Player 1, 1: Player 2
        self.first_player_of_round = self.current_player_turn # Player who starts the round or after a score reset
        print(f"第一回合先手: Player {self.current_player_turn + 1}")

        self.selected_object: Optional[GameObject] = None
        self.is_dragging = False
        self.drag_start_pos = None
        self.game_over = False
        self.winner = None
        self.action_processing_pending = False
        self.frame_count = 0 # 游戏帧计数器
        print("DEBUG: Game.__init__ finished") # DEBUG

    def setup_objects(self):
        # Player 1 (左方, 蓝色)
        for i in range(3):
            obj = GameObject(
                x=100,
                y=SCREEN_HEIGHT // 2 - 100 + i * 100,
                radius=OBJECT_RADIUS,
                color=BLUE,
                hp=INITIAL_HP,
                attack=INITIAL_ATTACK,
                player_id=0,
                object_id=i
            )
            self.players_objects[0].append(obj)

        # Player 2 (右方, 红色)
        for i in range(3):
            obj = GameObject(
                x=SCREEN_WIDTH - 100,
                y=SCREEN_HEIGHT // 2 - 100 + i * 100,
                radius=OBJECT_RADIUS,
                color=RED,
                hp=INITIAL_HP,
                attack=INITIAL_ATTACK,
                player_id=1,
                object_id=i
            )
            self.players_objects[1].append(obj)

    def all_objects_stopped(self):
        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.is_moving:
                    return False
        return True

    def current_player_all_moved(self):
        for obj in self.players_objects[self.current_player_turn]:
            if obj.hp > 0 and not obj.has_moved_this_turn: # 只有存活的单位才能行动
                return False
        return True

    def can_player_move(self, player_id: int) -> bool:
        """Checks if the specified player has any units that can still move."""
        for obj in self.players_objects[player_id]:
            if obj.hp > 0 and not obj.has_moved_this_turn:
                return True
        return False

    def switch_player(self):
        self.current_player_turn = 1 - self.current_player_turn

    def next_round(self):
        print("回合结束，交换先手")
        self.first_player_of_round = 1 - self.first_player_of_round
        self.current_player_turn = self.first_player_of_round
        for player_objs in self.players_objects:
            for obj in player_objs:
                obj.has_moved_this_turn = False # 重置所有单位的行动状态
        print(f"下一回合开始，Player {self.current_player_turn + 1} 先手")

    def check_object_destruction_and_scoring(self):
        if self.game_over: # Don't process if game already ended
            return

        player0_alive_units = sum(1 for obj in self.players_objects[0] if obj.hp > 0)
        player1_alive_units = sum(1 for obj in self.players_objects[1] if obj.hp > 0)

        point_scored_by_player = -1 # -1: no score, 0: player 0 scores, 1: player 1 scores

        if player0_alive_units == 0 and player1_alive_units > 0:
            self.scores[1] += 1
            point_scored_by_player = 1
            print(f"Player 2 scores a point! Score: P1 {self.scores[0]} - P2 {self.scores[1]}")
        elif player1_alive_units == 0 and player0_alive_units > 0:
            self.scores[0] += 1
            point_scored_by_player = 0
            print(f"Player 1 scores a point! Score: P1 {self.scores[0]} - P2 {self.scores[1]}")
        elif player0_alive_units == 0 and player1_alive_units == 0: # Mutual destruction
            print("Mutual destruction! No point scored. Resetting board.")
            point_scored_by_player = -2 # Special value for mutual destruction reset

        if point_scored_by_player != -1: # If a point was scored or mutual destruction
            if self.scores[0] >= MAX_VICTORY_POINTS:
                self.game_over = True
                self.winner = 0
                print("Player 1 wins the game!")
                return # Game over, no further reset needed for turns
            elif self.scores[1] >= MAX_VICTORY_POINTS:
                self.game_over = True
                self.winner = 1
                print("Player 2 wins the game!")
                return # Game over, no further reset needed for turns

            # Respawn all objects for both players and reset their moved status
            for player_idx in range(2):
                for obj in self.players_objects[player_idx]:
                    obj.respawn()
                    obj.has_moved_this_turn = False
            
            # After a score and reset, the turn goes to the designated first player of the current round.
            self.current_player_turn = self.first_player_of_round
            print(f"Board reset. Player {self.current_player_turn + 1} (round starter) to move.")
            return True # Indicates a score and reset occurred

        return False # Indicates no score/reset occurred

    def draw_drag_line(self):
        if self.is_dragging and self.selected_object and self.drag_start_pos:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            obj_center_x, obj_center_y = self.selected_object.x, self.selected_object.y

            dx_mouse = mouse_x - obj_center_x
            dy_mouse = mouse_y - obj_center_y

            launch_vx = -dx_mouse
            launch_vy = -dy_mouse

            line_length_factor = 0.5
            end_line_x = obj_center_x + launch_vx * line_length_factor
            end_line_y = obj_center_y + launch_vy * line_length_factor

            pygame.draw.line(self.screen, GREEN, (obj_center_x, obj_center_y), (end_line_x, end_line_y), 2)
            pygame.draw.circle(self.screen, GREEN, (int(end_line_x), int(end_line_y)), 5)

    def draw(self):
        self.screen.fill(WHITE)

        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.hp > 0:
                    obj.draw(self.screen)

        self.draw_drag_line()

        score_text_p1 = self.font.render(f"Player 1 (Blue): {self.scores[0]}", True, BLUE)
        score_text_p2 = self.font.render(f"Player 2 (Red): {self.scores[1]}", True, RED)
        self.screen.blit(score_text_p1, (10, 10))
        self.screen.blit(score_text_p2, (SCREEN_WIDTH - score_text_p2.get_width() - 10, 10))

        turn_indicator_color = BLUE if self.current_player_turn == 0 else RED
        player_name = "Player 1" if self.current_player_turn == 0 else "Player 2"

        action_prompt = ""
        if not self.game_over:
            current_player_can_move_now = False
            if self.all_objects_stopped():
                if self.can_player_move(self.current_player_turn):
                    current_player_can_move_now = True
                
                if current_player_can_move_now:
                    action_prompt = f"{player_name}, select an object to launch."
                else:
                    action_prompt = f"{player_name} has no more units to move this turn. Waiting..."
            else:
                action_prompt = "Objects are moving..."

        turn_text_surface = self.font.render(action_prompt, True, turn_indicator_color)
        turn_text_rect = turn_text_surface.get_rect(center=(SCREEN_WIDTH // 2, 30))
        self.screen.blit(turn_text_surface, turn_text_rect)

        moved_status_texts_surfaces = []
        y_offset_for_status = 50

        p1_label_surf = self.font.render(f"P1 Units:", True, BLACK)
        self.screen.blit(p1_label_surf, (10, y_offset_for_status))
        current_y_p1 = y_offset_for_status + 25
        for obj_idx, obj in enumerate(self.players_objects[0]):
            status = "Moved" if obj.has_moved_this_turn else "Ready"
            if obj.hp <=0: status = "KO"
            obj_text = f"  Obj {obj_idx+1}: {status}"
            text_surface = self.font.render(obj_text, True, BLACK)
            self.screen.blit(text_surface, (10, current_y_p1))
            current_y_p1 += 20

        p2_label_text = f"P2 Units:"
        p2_label_size_x, _ = self.font.size(p2_label_text)
        p2_label_surf = self.font.render(p2_label_text, True, BLACK)
        base_x_p2 = SCREEN_WIDTH - p2_label_size_x - 10
        self.screen.blit(p2_label_surf, (base_x_p2, y_offset_for_status))
        current_y_p2 = y_offset_for_status + 25
        for obj_idx, obj in enumerate(self.players_objects[1]):
            status = "Moved" if obj.has_moved_this_turn else "Ready"
            if obj.hp <=0: status = "KO"
            obj_text = f"  Obj {obj_idx+1}: {status}"
            text_surface = self.font.render(obj_text, True, BLACK)
            self.screen.blit(text_surface, (base_x_p2, current_y_p2))
            current_y_p2 += 20

        if self.game_over:
            winner_name = "Player 1" if self.winner == 0 else "Player 2"
            winner_color = BLUE if self.winner == 0 else RED
            game_over_text = self.font.render(f"Game Over! {winner_name} Wins!", True, winner_color)
            restart_text = self.font.render("Press R to Restart", True, BLACK)

            go_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
            rs_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))

            self.screen.blit(game_over_text, go_rect)
            self.screen.blit(restart_text, rs_rect)

        pygame.display.flip()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if self.game_over:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.__init__()
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.all_objects_stopped():
                    mouse_x, mouse_y = event.pos
                    for obj in self.players_objects[self.current_player_turn]:
                        if obj.hp > 0 and not obj.has_moved_this_turn:
                            distance = math.hypot(obj.x - mouse_x, obj.y - mouse_y)
                            if distance <= obj.radius:
                                self.selected_object = obj
                                self.is_dragging = True
                                self.drag_start_pos = (obj.x, obj.y)
                                break

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.is_dragging and self.selected_object:
                    current_selected_object = self.selected_object

                    mouse_x, mouse_y = event.pos
                    dx = current_selected_object.x - mouse_x
                    dy = current_selected_object.y - mouse_y

                    if current_selected_object.hp > 0 and not current_selected_object.has_moved_this_turn:
                        current_selected_object.apply_force(dx, dy)
                        self.action_processing_pending = True
                    self.is_dragging = False
                    self.selected_object = None

            if event.type == pygame.MOUSEMOTION:
                if self.is_dragging and self.selected_object:
                    current_selected_object = self.selected_object
                    mouse_x, mouse_y = event.pos
                    drag_dx_from_center = mouse_x - current_selected_object.x
                    drag_dy_from_center = mouse_y - current_selected_object.y
                    current_selected_object.angle = math.atan2(drag_dy_from_center, drag_dx_from_center)
        return True

    def update(self):
        if self.game_over:
            return

        any_object_moving_after_launch = False
        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.is_moving:
                    obj.move()
                    obj.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT)
                    any_object_moving_after_launch = True

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
                    if distance == 0: continue

                min_dist = obj1.radius + obj2.radius
                if distance < min_dist:
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

                            if obj1.player_id != obj2.player_id:
                                obj1.take_damage(obj2.attack, self.frame_count)
                                obj2.take_damage(obj1.attack, self.frame_count)

        if self.action_processing_pending and self.all_objects_stopped():
            self.action_processing_pending = False

            score_occurred_and_reset = self.check_object_destruction_and_scoring()

            if self.game_over:
                return

            if not score_occurred_and_reset:
                player_who_just_moved = self.current_player_turn
                potential_next_player = 1 - player_who_just_moved

                can_potential_next_player_move = self.can_player_move(potential_next_player)
                can_player_who_just_moved_still_move = self.can_player_move(player_who_just_moved)

                if can_potential_next_player_move:
                    self.current_player_turn = potential_next_player
                    print(f"Player {player_who_just_moved + 1} finished move. Now Player {self.current_player_turn + 1}'s turn.")
                elif can_player_who_just_moved_still_move:
                    print(f"Player {potential_next_player + 1} has no more moves. Player {self.current_player_turn + 1} continues.")
                else:
                    self.next_round()

# --- 主程序循环 ---
if __name__ == '__main__':
    game = Game()
    running = True
    while running:
        if not game.handle_input():
            running = False
            break

        game.update()
        game.draw()

        game.clock.tick(FPS)
        game.frame_count += 1

    pygame.quit()
