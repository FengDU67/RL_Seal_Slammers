# RL Seal Slammers

此项目灵感来自米哈游《崩坏：星穹铁道》中的“豹豹碰碰大作战”活动。

本项目通过自定义 Pygame 物理+Gymnasium 环境 + Stable-Baselines3 (sb3-contrib MaskablePPO) 来训练智能体策略，并提供人类游玩与可视化轨迹预测。

---
## 目录
1. 功能概述
2. 环境与依赖安装
3. 代码结构概览
4. 强化学习环境说明
5. 训练使用方法 (fresh / resume / init-from)
6. 奖励设计与 TensorBoard 指标
7. 运行可视化 / 对局(play)方式
8. 模型文件管理
9. 常见问题 (FAQ)
10. 贡献方式

---
## 1. 功能概述
- 多对象回合制“弹射+碰撞”对战玩法仿真。
- MultiDiscrete 动作空间: [选择己方对象, 72 个方向, 5 档力度]。
- Action Masking：只允许尚未行动且存活的己方单位被选中。
- 轨迹预测：拖拽时的虚线轨迹与环境离散化后的真实发射一致。
- 奖励 shaping 可分阶段衰减，并记录各组件分解。
- 支持：
  - 从头训练 (fresh)
  - 继续训练 (resume)
  - 使用已有模型权重作为初始化重新训练 (init-from)
- TensorBoard 记录：学习率、loss、策略熵、奖励组件等。
- 对手与自博弈：
  - Greedy 一步对手（单边训练用）
  - MCTS 对手（可接入 PPO 策略先验与价值评估，含 progressive widening 与根噪声）
  - 双边 MCTS 自我博弈环境，支持 z-only 终局奖励（AlphaZero 风格）

---
## 2. 环境与依赖安装
建议使用 Python 3.10+ (已在 3.12 上开发)。

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install stable-baselines3 sb3-contrib pygame numpy gymnasium tensorboard
```
如需 GPU：确保正确安装 CUDA + 对应版本 torch。

验证安装：
```bash
python -c "import torch;print(torch.cuda.is_available())"
```

---
## 3. 代码结构概览
```
main.py                          # (可存在) 示例/调试纯游戏入口
rl_seal_slammers/
  envs/seal_slammers_env.py      # 基础环境（双边轮流，保持完整奖励）
  envs/greedy_one_step.py        # 一步贪心对手
  envs/mcts_agent.py             # MCTS 对手 + PPOPolicyAdapter
  envs/seal_slammers_single_sided_greedy_env.py   # 单边训练：对手=Greedy
  envs/seal_slammers_single_sided_mcts_env.py     # 单边训练：对手=MCTS（可接 PPO 先验/价值）
  envs/seal_slammers_mcts_selfplay_env.py         # 双边 MCTS 自博弈（可选 z-only）
  physics_utils.py               # 共享物理/轨迹预测
  scripts/
    train.py                     # 训练入口 (MaskablePPO)
    play.py                      # 人类游玩 / 轨迹可视化 / RL 环境驱动
models/                          # 训练生成的模型
logs/                            # TensorBoard 日志 (含评估)
```

---
## 4. 强化学习环境说明
动作空间 (MultiDiscrete):
- object_idx: [0 .. num_objects_per_player-1]
- angle_idx: 72 等分方向 (0 对应 0 rad，发射角 = 方向角 + π)
- strength_idx: 5 档力度 (比例 = (idx+1)/5)

状态特征 (扁平向量):
- 每个对象 5 个特征: 相对位置 (x,y), 归一化 HP, 归一化攻击, 本回合是否已行动
- 全局: 当前行动玩家, 双方分数
- 上一回合对手动作 (对象/角度/力度 索引归一化)

主要物理参数：
- 摩擦 FRICTION=0.98
- 弹性系数 ELASTICITY=1.0 (全弹性)
- 速度阈值 |vx|,|vy|<0.1 即停止
- 发射强度离散化基于 OBJECT_RADIUS * 4 * FORCE_MULTIPLIER

---
## 5. 训练使用方法
进入项目根目录后执行。

### 5.1 从零开始 (Fresh)
```bash
python rl_seal_slammers/scripts/train.py \
  --target-total-timesteps 5000000 \
  --lr-initial 3e-4 --lr-final 5e-5 \
  --env base \
  --run-name 2025-08-15-exp1   # 可选，不写则自动用时间戳
```

单边 MCTS 对手：
```bash
python rl_seal_slammers/scripts/train.py \
  --env single_sided_mcts \
  --mcts-sims 128 --mcts-cpuct 1.4 --mcts-max-depth 3 \
  --mcts-angle-step 6 --mcts-strength-topk 2 \
  --run-name mcts-ss-128x
```

双边 MCTS 自博弈（AlphaZero 风格 z-only）：
```bash
python rl_seal_slammers/scripts/train.py \
  --env mcts_selfplay --z-only \
  --mcts-sims 160 --mcts-cpuct 1.25 --mcts-max-depth 3 \
  --mcts-angle-step 6 --mcts-strength-topk 2 \
  --run-name azero-sp-160x
```
输出:
- 日志: logs/sb3_maskable_ppo_sealslammers_mlp/<env>/<run-name>/
- 模型: models/sb3_maskable_ppo_sealslammers_mlp/<env>/<run-name>/maskable_ppo_sealslammers_mlp_model_final.zip
- 最优评估模型: models/.../<env>/<run-name>/maskable_ppo_sealslammers_mlp_model_best/best_model.zip

### 5.2 继续训练 (Resume)
保持当前优化器状态 + timesteps 连续。
```bash
python rl_seal_slammers/scripts/train.py \
  --resume-from models/sb3_maskable_ppo_sealslammers_mlp/maskable_ppo_sealslammers_mlp_model_final.zip \
  --target-total-timesteps 8000000
```
若已训练步数 >= 目标，会直接退出。

保存策略：
- 继续训练会在“运行专属目录”保存一份，同时覆盖 --resume-from 指定的原模型文件（你要求的“继续训练可以覆盖原来的模型”）。
- fresh/init-from 则总是保存到新的 run 目录，不会覆盖旧模型（“不要覆盖原来的模型”）。

### 5.3 以已有模型作为初始化重新训练 (Init From)
只加载网络权重，不加载 optimizer / timesteps；重新计数与调度。
```bash
python rl_seal_slammers/scripts/train.py \
  --init-from models/sb3_maskable_ppo_sealslammers_mlp/maskable_ppo_sealslammers_mlp_model_final.zip \
  --target-total-timesteps 5000000
```
适用于新 reward / 新超参但想用旧策略热启动。

### 5.4 参数说明
| 参数 | 说明 |
|------|------|
| --env | 训练环境：`base` / `single_sided_greedy` / `single_sided_mcts` / `mcts_selfplay` |
| --run-name | 运行目录名；fresh/init 未指定则用时间戳；resume 未指定则根据模型文件名推断 |
| --resume-from | 继续训练文件路径 (.zip) |
| --init-from | 作为初始化加载权重 |
| --target-total-timesteps | 目标全局步数 (resume: 只训练剩余; fresh/init: 全量) |
| --lr-initial / --lr-final | 线性学习率调度起止值 |
| --z-only | 仅 mcts_selfplay 使用：对 PPO 暴露 ±1/0 终局奖励 |
| --mcts-sims | MCTS 每步仿真次数（单边/自博弈） |
| --mcts-cpuct | PUCT 探索常数 |
| --mcts-max-depth | 搜索最大深度 |
| --mcts-angle-step | 角度离散步长（例如 6 表示每 6° 取一个候选） |
| --mcts-strength-topk | 每个对象考虑前 K 档力度 |

提示：训练脚本会在 MCTS 环境中自动注入 PPOPolicyAdapter，使 MCTS 在扩展与价值回传时使用当前 PPO 的策略先验与价值评估。

二者不可同时使用: resume 与 init-from 互斥。

### 5.5 启动 TensorBoard
```bash
tensorboard --logdir logs/sb3_maskable_ppo_sealslammers_mlp --port 6006
```
浏览 http://localhost:6006

---
## 6. 奖励设计与 TensorBoard 指标
环境 step 返回时内部记录以下组件并在 episode 终止时聚合：
- ko_points (击杀得分)
- win_bonus / loss_penalty
- invalid_penalty (非法/无效选择)
- selection_reward (选取有效对象)
- damage_base (线性伤害)
- tier_bonus (伤害阈值奖励)
- meaningless_penalty (连续无效行动惩罚)
- collision_reward (无伤碰撞)
- dist_reward (缩短距离)
- alignment_reward (朝向目标)
- position_reward (额外位置 shaping)
- time_penalty (时间步微小惩罚)
- total_reward (汇总)

TensorBoard 中记录: reward_components/<name>
以及 reward_components/shaping_scale 追踪 shaping 衰减。

注：在 `mcts_selfplay` 环境并开启 `--z-only` 时，PPO 接收到的是 z-only 奖励（±1/0），不再包含上述 shaping 组件；但游戏内部仍按完整规则判定胜负与物理结算。

---
## 7. 游玩 / 可视化 (Play)
脚本: `rl_seal_slammers/scripts/play.py`

支持模式 (参数 `--mode`):
- `human_vs_human` : 双人本地轮流操作
- `human_vs_ai`    : 玩家(蓝方, Player 1) 对 AI(红方, Player 2)
- `ai_vs_human`    : AI(蓝方) 对 玩家(红方)
- `ai_vs_ai`       : 双方均为 AI (有人类渲染窗口)
- `ai_vs_ai_fast`  : 双方 AI，取消实时渲染（加速模拟）

可选主要参数:
| 参数 | 说明 | 示例 |
|------|------|------|
| `--mode` | 对战模式 | `--mode human_vs_ai` |
| `--num-objects` | 每方棋子数量 (默认 3) | `--num-objects 4` |
| `--model-path` | 指定 PPO 模型路径 (MaskablePPO/PPO) | `--model-path models/.../best_model.zip` |
| `--model-env` | 不给路径时，用此 env（base/single_sided_greedy/single_sided_mcts/mcts_selfplay）下最新 run 自动查找 | `--model-env base` |
| `--run-name` | 配合 `--model-env` 指定具体 run 目录 | `--run-name 2025-08-15-exp1` |
| `--p0-hp` / `--p1-hp` | 固定双方初始 HP (默认环境内部值) | `--p0-hp 45 --p1-hp 50` |
| `--p0-atk` / `--p1-atk` | 固定双方初始 ATK | `--p0-atk 9 --p1-atk 8` |

基础运行 (本地人人):
```bash
python rl_seal_slammers/scripts/play.py --mode human_vs_human
```
玩家 vs 训练模型（自动从 base 环境下最新 run 取 final 模型）:
```bash
python rl_seal_slammers/scripts/play.py --mode human_vs_ai --model-env base
```
指定具体 run：
```bash
python rl_seal_slammers/scripts/play.py --mode human_vs_ai --model-env base --run-name 2025-08-15-exp1
```
模型 vs 玩家(玩家操作红方):
```bash
python rl_seal_slammers/scripts/play.py --mode ai_vs_human --model models/.../best_model.zip
```
模型自博弈 (渲染):
```bash
python rl_seal_slammers/scripts/play.py --mode ai_vs_ai --model models/.../best_model.zip
```
模型自博弈 (快速，无窗口):
```bash
python rl_seal_slammers/scripts/play.py --mode ai_vs_ai_fast --model models/.../best_model.zip
```

---
## 8. 模型文件管理
保存布局（新）：
```
models/sb3_maskable_ppo_sealslammers_mlp/
  <env>/
    <run-name>/
      maskable_ppo_sealslammers_mlp_model_final.zip
      maskable_ppo_sealslammers_mlp_model_best/
        best_model.zip
```
日志布局：
```
logs/sb3_maskable_ppo_sealslammers_mlp/
  <env>/<run-name>/
```

建议额外手动加：
- 按时间戳归档 (避免覆盖)
- 保存中间 checkpoint (可扩展自定义 Callback)

扩展 checkpoint 思路：
- 自定义 BaseCallback，定期调用 model.save(f".../ckpt_{self.num_timesteps}.zip")。

---
## 9. 常见问题 (FAQ)
Q: 训练很慢？
A: 降低 n_steps 或 batch_size、减少并行环境数，或关闭渲染。确认未在 debug 模式中频繁打印。

Q: eval reward 与训练 ep_rew_mean 差异大？
A: 确保 eval 环境同步 shaping_scale（当前 train.py 已同步），并关注 shaping 削弱期进行的指标变化。

Q: 想快速测试策略？
A: 使用 --target-total-timesteps 较小值(如 2e5) + 减少网络规模。

Q: init-from 与 resume 区别？
A: resume 继续优化器状态与时间步；init-from 仅迁移网络权重，学习率调度与计步从 0 重新开始。

Q: 有多个环境 (base / single_sided_greedy) 时如何区分保存？
A: 训练脚本会按 `models/.../<env>/<run-name>/` 和 `logs/.../<env>/<run-name>/` 分类保存。fresh/init-from 不覆盖旧模型；resume 会同时覆盖 `--resume-from` 指定文件。`--env` 现支持 `single_sided_mcts` 与 `mcts_selfplay`，可和 base/greedy 一起对比实验。

---
## 10. 贡献方式
欢迎提交: Issue / PR / 优化建议 / 新的奖励设计。请尽量：
- 说明改动动机
- 附带复现步骤
- 遵守现有代码风格

---
## 许可证
(如果需要请在此添加 License 声明，如 MIT / Apache-2.0)。

----
如需：
- 增加对战模式 (人类 vs 训练策略)
- 增加多模型对战评估脚本
- 增加分布式训练 / 自适应 curriculum
请提出 Issue 或继续交流。