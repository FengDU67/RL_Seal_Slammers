import os
import sys
import torch  # Added import
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy # CHANGED POLICY IMPORT
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env  # Changed import
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

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
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "sb3_ppo_sealslammers_mlp") # CHANGED
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "sb3_ppo_sealslammers_mlp") # CHANGED
MODEL_SAVE_NAME = "ppo_sealslammers_mlp_model" # CHANGED

TOTAL_TIMESTEPS = 500000  # Total training timesteps
N_ENVS = 16                 # Number of parallel environments
MODEL_SAVE_FREQ = 50000    # Save a checkpoint every N total steps

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train_agent():
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"Model Directory: {MODEL_DIR}")

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
    env = make_vec_env(SealSlammersEnv, n_envs=N_ENVS, env_kwargs=env_kwargs) # Removed vec_env_cls override
    print("Vectorized environment created.")

    print("Initializing PPO agent with MlpPolicy...") # CHANGED MESSAGE
    # policy_kwargs for MlpPolicy (if needed, e.g., for net_arch)
    # For default MlpPolicy architecture, policy_kwargs can be None or an empty dict.
    # Example: policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    # Removing LSTM-specific kwargs
    policy_kwargs = {} # Or None. Using {} for now.

    model = PPO(
        MlpPolicy,  # CHANGED POLICY CLASS
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs, # UPDATED policy_kwargs
        n_steps=512,         # CHANGED: Number of steps to run for each environment per update
        batch_size=64,       
        n_epochs=10,         
        gamma=0.99,          
        gae_lambda=0.95,     
        clip_range=0.2,      
        ent_coef=0.0,        
        vf_coef=0.5,         
        max_grad_norm=0.5,   
        learning_rate=3e-4,  
        device=device,       
        # seed=42,           
    )

    # Setup a callback to save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=MODEL_SAVE_FREQ, # Save based on total timesteps
        save_path=MODEL_DIR,
        name_prefix=MODEL_SAVE_NAME
    )

    print("Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        log_interval=1, # Log training info every N rollouts (N = N_ENVS * n_steps timesteps)
        # reset_num_timesteps=False # Set to False if you want to continue training from a loaded model
    )

    # Save the final model
    final_model_path = os.path.join(MODEL_DIR, f"{MODEL_SAVE_NAME}_final")
    model.save(final_model_path)
    print(f"Training finished. Final model saved to {final_model_path}.zip")

    env.close()
    print("Training script finished.")

if __name__ == "__main__":
    # To run this script:
    # 1. Navigate to the project root directory: cd /home/wlq/PycharmProjects/RL_Seal_Slammers
    # 2. Execute: python rl_seal_slammers/scripts/train.py
    train_agent()
