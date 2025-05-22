import pygame
import random
import math

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

        self.players_objects = [[], []] # 存储两方玩家的object
        self.scores = [0, 0] # 存储两方玩家的胜利点

        self.setup_objects()

        self.current_player_turn = random.choice([0, 1]) # 0: Player 1, 1: Player 2
        self.first_player_of_round = self.current_player_turn
        print(f"第一回合先手: Player {self.current_player_turn + 1}") # This is an existing print

        self.selected_object = None
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

    def switch_player(self):
        self.current_player_turn = 1 - self.current_player_turn
        print(f"轮到 Player {self.current_player_turn + 1} 行动")

    def next_round(self):
        print("回合结束，交换先手")
        self.first_player_of_round = 1 - self.first_player_of_round
        self.current_player_turn = self.first_player_of_round
        for player_objs in self.players_objects:
            for obj in player_objs:
                obj.has_moved_this_turn = False # 重置所有单位的行动状态
        print(f"下一回合开始，Player {self.current_player_turn + 1} 先手")


    def draw_drag_line(self):
        if self.is_dragging and self.selected_object and self.drag_start_pos:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # 从物体中心画线到鼠标当前位置的反方向预览发射方向和力度
            obj_center_x, obj_center_y = self.selected_object.x, self.selected_object.y

            # 向量: 从物体中心到鼠标 (dx, dy)
            dx_mouse = mouse_x - obj_center_x
            dy_mouse = mouse_y - obj_center_y

            # 反向向量，代表发射方向
            launch_vx = -dx_mouse
            launch_vy = -dy_mouse

            # 指示线终点 (可以根据力度调整长度)
            line_length_factor = 0.5 # 可调整，让线不要太长
            end_line_x = obj_center_x + launch_vx * line_length_factor
            end_line_y = obj_center_y + launch_vy * line_length_factor

            pygame.draw.line(self.screen, GREEN, (obj_center_x, obj_center_y), (end_line_x, end_line_y), 2)
            pygame.draw.circle(self.screen, GREEN, (int(end_line_x), int(end_line_y)), 5) # 箭头


    def draw(self):
        self.screen.fill(WHITE)

        # 绘制所有对象
        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.hp > 0: # 只绘制存活的物体
                    obj.draw(self.screen)

        # 绘制拖拽指示线
        self.draw_drag_line()

        # 显示分数和当前回合信息
        score_text_p1 = self.font.render(f"Player 1 (Blue): {self.scores[0]}", True, BLUE)
        score_text_p2 = self.font.render(f"Player 2 (Red): {self.scores[1]}", True, RED)
        self.screen.blit(score_text_p1, (10, 10))
        self.screen.blit(score_text_p2, (SCREEN_WIDTH - score_text_p2.get_width() - 10, 10))

        turn_indicator_color = BLUE if self.current_player_turn == 0 else RED
        player_name = "Player 1" if self.current_player_turn == 0 else "Player 2"

        action_prompt = ""
        if not self.game_over:
            current_player_can_move = False
            if self.all_objects_stopped(): # Ensure we check this only when objects are stopped
                for obj_ in self.players_objects[self.current_player_turn]:
                    if obj_.hp > 0 and not obj_.has_moved_this_turn:
                        current_player_can_move = True
                        break
                if current_player_can_move:
                    action_prompt = f"{player_name}, select an object to launch."
                else:
                    action_prompt = f"{player_name} has no more units to move. Waiting..."
            else:
                action_prompt = "Objects are moving..."


        turn_text_surface = self.font.render(action_prompt, True, turn_indicator_color)
        turn_text_rect = turn_text_surface.get_rect(center=(SCREEN_WIDTH // 2, 30))
        self.screen.blit(turn_text_surface, turn_text_rect)

        # 显示哪个单位已行动
        moved_status_texts_surfaces = []
        y_offset_for_status = 50 # Starting Y for the status block

        # Player 1 statuses
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

        # Player 2 statuses
        p2_label_text = f"P2 Units:"
        p2_label_size_x, _ = self.font.size(p2_label_text)
        p2_label_surf = self.font.render(p2_label_text, True, BLACK)
        base_x_p2 = SCREEN_WIDTH - p2_label_size_x - 10 # Align to right
        self.screen.blit(p2_label_surf, (base_x_p2, y_offset_for_status))
        current_y_p2 = y_offset_for_status + 25
        for obj_idx, obj in enumerate(self.players_objects[1]):
            status = "Moved" if obj.has_moved_this_turn else "Ready"
            if obj.hp <=0: status = "KO"
            obj_text = f"  Obj {obj_idx+1}: {status}"
            # To align text under P2 label, calculate its width or use a fixed offset from base_x_p2
            text_surface = self.font.render(obj_text, True, BLACK)
            self.screen.blit(text_surface, (base_x_p2, current_y_p2)) # Simple alignment under label
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
                return False # 结束游戏循环

            if self.game_over:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r: # 按R重新开始
                    self.__init__() # 重新初始化游戏
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # 左键按下
                if self.all_objects_stopped(): # 只有所有物体都停止了才能操作
                    mouse_x, mouse_y = event.pos
                    for obj in self.players_objects[self.current_player_turn]:
                        if obj.hp > 0 and not obj.has_moved_this_turn: # 只能选择自己未行动且存活的单位
                            distance = math.hypot(obj.x - mouse_x, obj.y - mouse_y)
                            if distance <= obj.radius:
                                self.selected_object = obj
                                self.is_dragging = True
                                self.drag_start_pos = (obj.x, obj.y) # 以物体中心为拖拽起点
                                break

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1: # 左键松开
                if self.is_dragging and self.selected_object:
                    current_selected_object = self.selected_object # Assign to local variable

                    mouse_x, mouse_y = event.pos
                    # 计算拖拽向量 (从物体中心到鼠标松开位置的反方向)
                    dx = current_selected_object.x - mouse_x
                    dy = current_selected_object.y - mouse_y

                    # 施加力，让物体弹射
                    if current_selected_object.hp > 0 and not current_selected_object.has_moved_this_turn: # 再次确认
                        current_selected_object.apply_force(dx, dy)
                        self.action_processing_pending = True # 标记有动作发生，待处理
                    self.is_dragging = False
                    self.selected_object = None # 操作完成后取消选择

            if event.type == pygame.MOUSEMOTION:
                if self.is_dragging and self.selected_object:
                    current_selected_object = self.selected_object # Assign to local variable
                    mouse_x, mouse_y = event.pos
                    # Update object's angle to point from its center to the mouse cursor
                    drag_dx_from_center = mouse_x - current_selected_object.x
                    drag_dy_from_center = mouse_y - current_selected_object.y
                    current_selected_object.angle = math.atan2(drag_dy_from_center, drag_dx_from_center)
                    # The drag line will be drawn based on this angle or mouse pos separately
        return True

    def update(self):
        if self.game_over:
            return

        any_object_moving_after_launch = False
        for player_objs in self.players_objects:
            for obj in player_objs:
                if obj.is_moving: # 只有移动中的物体才更新
                    obj.move()
                    obj.check_boundary_collision(SCREEN_WIDTH, SCREEN_HEIGHT)
                    any_object_moving_after_launch = True

        # 碰撞检测与处理
        all_objects = self.players_objects[0] + self.players_objects[1]
        for i in range(len(all_objects)):
            for j in range(i + 1, len(all_objects)):
                obj1 = all_objects[i]
                obj2 = all_objects[j]

                if obj1.hp <= 0 or obj2.hp <= 0: continue # 跳过已淘汰的物体

                dist_x = obj1.x - obj2.x
                dist_y = obj1.y - obj2.y
                distance = math.hypot(dist_x, dist_y)

                if distance == 0: # Avoid division by zero if objects are exactly at the same position
                    # Slightly move one object to prevent being stuck
                    obj1.x += random.uniform(-0.1, 0.1)
                    obj1.y += random.uniform(-0.1, 0.1)
                    dist_x = obj1.x - obj2.x
                    dist_y = obj1.y - obj2.y
                    distance = math.hypot(dist_x, dist_y)
                    if distance == 0: continue # Still zero, skip this pair for this frame


                if distance < obj1.radius + obj2.radius: # 发生碰撞
                    # 法向量 (points from obj2 to obj1)
                    nx = dist_x / distance
                    ny = dist_y / distance

                    # 相对速度在法线方向上的投影
                    rvx = obj1.vx - obj2.vx
                    rvy = obj1.vy - obj2.vy
                    vel_along_normal = rvx * nx + rvy * ny

                    if vel_along_normal < 0: # 只有当物体相互靠近时才处理碰撞反弹
                        # 使用冲量法处理碰撞 (考虑质量和弹性)
                        e = (obj1.restitution)

# --- 主程序循环 ---
if __name__ == '__main__':
    game = Game()
    running = True
    while running:
        if not game.handle_input(): # handle_input 返回 False 时退出循环
            running = False
            break

        game.update()
        game.draw()

        game.clock.tick(FPS)
        game.frame_count += 1 # 更新游戏帧计数

    pygame.quit()
