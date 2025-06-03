import os
import sys
import torch  # Added import
from sb3_contrib import MaskablePPO 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # SubprocVecEnv 用于并行化

# --- Project Root and Path Setup ---
# Assuming the script is in RL_Seal_Slammers/rl_seal_slammers/scripts/
# Project root is two levels up from the script directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add project root to sys.path to allow imports like 'from rl_seal_slammers.envs...'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now import the environment
from rl_seal_slammers.envs.seal_slammers_env import SealSlammersEnv

# --- Parameters ---
NUM_OBJECTS_PER_PLAYER = 3
TOTAL_TIMESTEPS = 2_000_000  # 增加训练步数
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "sb3_maskable_ppo_sealslammers_mlp")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "sb3_maskable_ppo_sealslammers_mlp")
MODEL_NAME_PREFIX = "maskable_ppo_sealslammers_mlp_model"
EVAL_FREQ = 50000  # 增加评估频率
N_EVAL_EPISODES = 10 # 增加评估episode数
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME_PREFIX}_best")

# 减少并行环境数量以获得更稳定的训练
N_ENVS = 8 # 从16减少到8，减少环境间差异

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_SAVE_PATH, exist_ok=True)

def train_agent():
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"Model Directory: {MODEL_SAVE_DIR}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Initializing SealSlammersEnv for check...")
    # render_mode=None is default, suitable for headless check
    single_env_instance = SealSlammersEnv(num_objects_per_player=NUM_OBJECTS_PER_PLAYER)
    print("Checking environment compatibility with SB3...")
    check_env(single_env_instance, warn=True, skip_render_check=True)
    print("Environment check passed.")
    single_env_instance.close()

    print(f"Creating vectorized environment with n_envs={N_ENVS}...")
    env_kwargs = {'num_objects_per_player': NUM_OBJECTS_PER_PLAYER}
    # For MlpPolicy, make_vec_env will use SubprocVecEnv by default if n_envs > 1
    # and DummyVecEnv if n_envs = 1.
    env_fns = [lambda: Monitor(SealSlammersEnv(render_mode=None)) for _ in range(N_ENVS)]
    env = SubprocVecEnv(env_fns)
    print(f"Using {N_ENVS} parallel environments.")

    print("Initializing PPO agent with MlpPolicy...") # CHANGED MESSAGE
    policy_kwargs = {} # Or None. Using {} for now.

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,       # 降低学习率
        n_steps=2048 // N_ENVS,   # 增加每个环境的步数
        batch_size=128,           # 增加batch size
        n_epochs=4,               # 减少epochs避免过拟合
        gamma=0.99,               
        gae_lambda=0.95,          
        ent_coef=0.01,            # 保持探索性
        vf_coef=0.5,              
        max_grad_norm=0.5,        
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(
            net_arch=[256, 256],  # 增加网络容量
            activation_fn=torch.nn.ReLU  # 明确指定激活函数
        )
    )
    print("MaskablePPO model created.")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # --- Callbacks ---
    # 评估回调，用于在训练过程中定期评估模型并保存最佳模型
    # Ensure eval_env is also a VecEnv, and preferably of the same type as the training env.
    # The training env is SubprocVecEnv.
    print("Creating evaluation environment...")
    eval_env_fns = [lambda: Monitor(SealSlammersEnv(num_objects_per_player=NUM_OBJECTS_PER_PLAYER, render_mode=None))]
    eval_env = SubprocVecEnv(eval_env_fns) # Use SubprocVecEnv for eval_env as well
    print(f"Evaluation environment type: {type(eval_env)}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1), # 调整评估频率以适应并行环境
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    # 可选：如果达到某个奖励阈值就停止训练的回调
    # reward_threshold = 200 # 示例阈值
    # stop_training_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    # combined_callback = [eval_callback, stop_training_callback]

    # --- Training ---
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback, # 或者 combined_callback
            progress_bar=True
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Save Final Model ---
        final_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME_PREFIX}_final.zip")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        print(f"Best model saved to {BEST_MODEL_SAVE_PATH}")
        print(f"Tensorboard logs available at: {LOG_DIR}")

    env.close() # 关闭环境
    print("Training script finished.")

if __name__ == "__main__":
    # 可选：检查环境是否符合 Gym API (主要用于调试环境)
    # print("Checking custom environment...")
    # check_env(SealSlammersEnv(render_mode=None)) 
    # print("Environment check passed.")
    
    train_agent()
