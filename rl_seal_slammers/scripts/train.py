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
TOTAL_TIMESTEPS = 1_000_000  # 示例：增加训练步数
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "sb3_maskable_ppo_sealslammers_mlp")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "sb3_maskable_ppo_sealslammers_mlp")
MODEL_NAME_PREFIX = "maskable_ppo_sealslammers_mlp_model"
EVAL_FREQ = 25000  # 每隔多少步评估一次
N_EVAL_EPISODES = 5 # 评估时运行多少个 episode
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME_PREFIX}_best")

# 并行环境数量 (如果使用 SubprocVecEnv)
N_ENVS = 16 # 示例：使用4个并行环境，可以根据您的CPU核心数调整

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
        learning_rate=3e-4,       # 示例值
        n_steps=512 // N_ENVS,   # 每个并行环境的步数，总步数 n_steps * N_ENVS
        batch_size=64,            # 示例值
        n_epochs=10,              # 示例值
        gamma=0.99,               # 示例值
        gae_lambda=0.95,          # 示例值
        ent_coef=0.01,            # 示例值，可以调整以平衡探索和利用
        vf_coef=0.5,              # 示例值
        max_grad_norm=0.5,        # 示例值
        verbose=1,
        tensorboard_log=LOG_DIR
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
