# Self-Play Training Script for Seal Slammers
# 自对弈训练脚本

import os
import sys
import torch
import numpy as np
from sb3_contrib import MaskablePPO 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# --- Project Root and Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_seal_slammers.envs.seal_slammers_env import SealSlammersEnv

class SelfPlayCallback(BaseCallback):
    """
    自对弈回调：定期更新对手模型
    """
    def __init__(self, update_freq=100000, verbose=1):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.opponent_model = None
        
    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            if self.verbose > 0:
                print(f"Updating opponent model at step {self.n_calls}")
            # 复制当前模型作为对手
            self.opponent_model = self.model.policy.clone()
        return True

class CurriculumEnv(SealSlammersEnv):
    """
    课程学习环境：逐步增加难度
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_stage = 0
        self.episode_count = 0
        
    def reset(self, seed=None, options=None):
        self.episode_count += 1
        
        # 课程学习：逐步增加HP和攻击力的随机性
        if self.episode_count < 1000:
            # 第一阶段：固定较低的HP和攻击力
            self.p0_hp_fixed = 30
            self.p0_atk_fixed = 6
            self.p1_hp_fixed = 30
            self.p1_atk_fixed = 6
        elif self.episode_count < 3000:
            # 第二阶段：少量随机性
            self.hp_range = (28, 32)
            self.atk_range = (5, 7)
            self.p0_hp_fixed = None
            self.p0_atk_fixed = None
            self.p1_hp_fixed = None
            self.p1_atk_fixed = None
        else:
            # 第三阶段：完全随机
            self.hp_range = (25, 50)
            self.atk_range = (4, 12)
            
        return super().reset(seed, options)

def train_with_curriculum():
    print("开始课程学习训练...")
    
    # 参数
    TOTAL_TIMESTEPS = 3_000_000
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "curriculum_training")
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "curriculum_training")
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 创建课程学习环境
    env = DummyVecEnv([lambda: Monitor(CurriculumEnv(render_mode=None))])
    
    # 创建模型
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,  # 稍微增加探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(
            net_arch=[128, 128],
            activation_fn=torch.nn.ReLU
        )
    )
    
    # 自对弈回调
    selfplay_callback = SelfPlayCallback(update_freq=200000)
    
    # 评估环境
    eval_env = DummyVecEnv([lambda: Monitor(SealSlammersEnv(render_mode=None))])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, "best_model"),
        log_path=LOG_DIR,
        eval_freq=50000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print(f"开始训练 {TOTAL_TIMESTEPS} 步...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[selfplay_callback, eval_callback],
        progress_bar=True
    )
    
    # 保存最终模型
    final_model_path = os.path.join(MODEL_SAVE_DIR, "curriculum_final.zip")
    model.save(final_model_path)
    print(f"训练完成！模型已保存到 {final_model_path}")
    
    env.close()

if __name__ == "__main__":
    train_with_curriculum()
