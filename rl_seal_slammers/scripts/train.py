import os
import sys
import torch
import argparse
from sb3_contrib import MaskablePPO 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from sb3_contrib.common.wrappers import ActionMasker

# --- Project Root and Path Setup ---
# Assuming the script is in RL_Seal_Slammers/rl_seal_slammers/scripts/
# Project root is two levels up from the script directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add project root to sys.path to allow imports like 'from rl_seal_slammers.envs...'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now import the environment
from rl_seal_slammers.envs.seal_slammers_env import SealSlammersEnv

# --- Helpers ---

def make_linear_schedule(initial_value: float, final_value: float):
    """Returns a schedule function for SB3 that linearly anneals from initial to final.
    SB3 passes progress_remaining in [1.0 -> 0.0].
    """
    def schedule(progress_remaining: float) -> float:
        return float(final_value + (initial_value - final_value) * progress_remaining)
    return schedule

class EntropyPhaseCallback(BaseCallback):
    """Two-phase entropy: hold high until hold_ratio, then linear decay to end_coef."""
    def __init__(self, start_coef: float, end_coef: float, total_timesteps: int, hold_ratio: float = 0.4, verbose: int = 0):
        super().__init__(verbose)
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.total_timesteps = total_timesteps
        self.hold_steps = int(total_timesteps * hold_ratio)

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t <= self.hold_steps:
            current = self.start_coef
        else:
            tail = self.total_timesteps - self.hold_steps
            if tail <= 0:
                current = self.end_coef
            else:
                prog = min(max(t - self.hold_steps, 0) / tail, 1.0)
                current = self.start_coef + (self.end_coef - self.start_coef) * prog
        try:
            self.model.ent_coef = float(current)
        except Exception:
            pass
        return True

class ShapingDecayCallback(BaseCallback):
    """Decay env.shaping_scale across timesteps (piecewise)."""
    def __init__(self, total_timesteps: int, milestones=(0.5, 0.8), scales=(1.0, 0.6, 0.3), verbose: int = 0):
        super().__init__(verbose)
        self.total = total_timesteps
        self.ms = milestones
        self.scales = scales

    def _set_scale(self, env):
        ratio = self.num_timesteps / self.total if self.total > 0 else 1.0
        if ratio < self.ms[0]:
            scale = self.scales[0]
        elif ratio < self.ms[1]:
            scale = self.scales[1]
        else:
            scale = self.scales[2]
        if hasattr(env, 'envs'):
            for e in env.envs:
                base = getattr(e, 'env', e)
                if hasattr(base, 'set_shaping_scale'):
                    base.set_shaping_scale(scale)
        elif hasattr(env, 'set_shaping_scale'):
            env.set_shaping_scale(scale)

    def _on_step(self) -> bool:
        self._set_scale(self.training_env)
        return True

# NEW: action mask function for ActionMasker

def mask_fn(env):
    # Env defines action_masks(); wrap base env to provide masks to MaskablePPO
    return env.action_masks()

# --- Parameters ---
NUM_OBJECTS_PER_PLAYER = 3
TOTAL_TIMESTEPS = 5_000_000  # default target total timesteps (global)
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "sb3_maskable_ppo_sealslammers_mlp")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "sb3_maskable_ppo_sealslammers_mlp")
MODEL_NAME_PREFIX = "maskable_ppo_sealslammers_mlp_model"
EVAL_FREQ = 80000
N_EVAL_EPISODES = 15
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME_PREFIX}_best")
N_ENVS = 12  # 从 8 提升到 12

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_SAVE_PATH, exist_ok=True)

def train_agent(resume_from=None, init_from=None, target_total_timesteps=TOTAL_TIMESTEPS, lr_initial=3e-4, lr_final=5e-5):
    if resume_from and init_from:
        print("Cannot use both --resume-from and --init-from simultaneously.")
        return
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
    # Wrap training envs: ActionMasker -> TimeLimit -> Monitor
    env_fns = [
        lambda: Monitor(
            TimeLimit(
                ActionMasker(SealSlammersEnv(render_mode=None, **env_kwargs), mask_fn),
                max_episode_steps=300
            )
        )
        for _ in range(N_ENVS)
    ]
    env = SubprocVecEnv(env_fns)
    print(f"Using {N_ENVS} parallel environments.")

    def create_fresh_model():
        return MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=make_linear_schedule(lr_initial, lr_final),
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.03,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=LOG_DIR,
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=torch.nn.ReLU
            )
        )

    # Decide model creation path
    model = None
    start_global_steps = 0
    target_total = int(target_total_timesteps)
    remaining = target_total

    if resume_from and os.path.isfile(resume_from):
        print(f"Resuming (continuing) from existing model: {resume_from}")
        try:
            model = MaskablePPO.load(resume_from, env=env, device=device, tensorboard_log=LOG_DIR)
            start_global_steps = model.num_timesteps
            if start_global_steps >= target_total:
                print(f"Existing model already at {start_global_steps} >= target {target_total}. Nothing to do.")
                env.close(); return
            remaining = target_total - start_global_steps
            print(f"Continue training: have {start_global_steps}, need {target_total}, remaining {remaining}")
        except Exception as e:
            print(f"Failed to load resume model: {e}. Starting fresh.")
            model = None
    elif init_from and os.path.isfile(init_from):
        print(f"Initializing weights from pretrained model (fresh run): {init_from}")
        try:
            pretrained = MaskablePPO.load(init_from, device=device)
            # Build fresh model (new optimizer, lr schedule, timesteps reset)
            model = create_fresh_model()
            # Sanity check spaces
            assert model.observation_space == env.observation_space
            assert model.action_space == env.action_space
            model.set_parameters(pretrained.get_parameters(), exact_match=True)
            print("Weights loaded into fresh model. Timesteps reset to 0.")
        except Exception as e:
            print(f"Failed to initialize from {init_from}: {e}. Falling back to fresh init.")
            model = None
    
    if model is None:
        if not resume_from:
            print("Initializing PPO agent with MlpPolicy (fresh training)...")
        model = create_fresh_model()
        print("MaskablePPO model created.")
    else:
        if resume_from:
            print("Loaded model; continuing training without resetting timesteps.")
        elif init_from:
            # Ensure num_timesteps is zero for fresh logging / schedules
            model.num_timesteps = 0

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # --- Callbacks ---
    print("Creating evaluation environment...")
    eval_env_fns = [
        lambda: Monitor(
            TimeLimit(
                ActionMasker(SealSlammersEnv(num_objects_per_player=NUM_OBJECTS_PER_PLAYER, render_mode=None), mask_fn),
                max_episode_steps=300
            )
        )
    ]
    eval_env = SubprocVecEnv(eval_env_fns)  # Use SubprocVecEnv for eval_env as well
    print(f"Evaluation environment type: {type(eval_env)}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=False,
        render=False
    )

    class EvalShapingSyncCallback(ShapingDecayCallback):
        def _on_step(self) -> bool:
            super()._on_step(); self._set_scale(eval_env); return True

    entropy_cb = EntropyPhaseCallback(start_coef=0.03, end_coef=0.007, total_timesteps=target_total, hold_ratio=0.4)
    shaping_cb = EvalShapingSyncCallback(total_timesteps=target_total)

    class RewardComponentTensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
        def _on_step(self) -> bool:
            infos = self.locals.get('infos', None)
            if not infos: return True
            acc, n = {}, 0
            for info in infos:
                if isinstance(info, dict):
                    comps = info.get('episode_reward_components')
                    if comps:
                        n += 1
                        for k,v in comps.items(): acc[k] = acc.get(k,0.0)+float(v)
            if n>0:
                for k,total_v in acc.items(): self.logger.record(f'reward_components/{k}', total_v/n)
                try:
                    env0 = self.training_env
                    scale=None
                    if hasattr(env0,'envs') and env0.envs:
                        base=getattr(env0.envs[0],'env', env0.envs[0])
                        if hasattr(base,'shaping_scale'): scale=base.shaping_scale
                    if scale is not None: self.logger.record('reward_components/shaping_scale', scale)
                except Exception: pass
            return True

    reward_comp_logger_cb = RewardComponentTensorboardCallback()
    callbacks = CallbackList([eval_callback, entropy_cb, shaping_cb, reward_comp_logger_cb])

    # Determine learn params
    if resume_from:
        total_timesteps_arg = remaining
        reset_flag = False
    else:
        # fresh or init_from (fresh timeline)
        total_timesteps_arg = target_total
        reset_flag = True

    print(f"Starting training. total_timesteps argument: {total_timesteps_arg}; reset_num_timesteps={reset_flag}")
    try:
        model.learn(
            total_timesteps=total_timesteps_arg,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_flag
        )
    except Exception as e:
        print(f"Training error: {e}")
        import traceback; traceback.print_exc()
    finally:
        # --- Save Final Model ---
        final_model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME_PREFIX}_final.zip")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        print(f"Best model saved to {BEST_MODEL_SAVE_PATH}")
        print(f"Tensorboard logs available at: {LOG_DIR}")
        # Close eval env (important to terminate subprocesses)
        try:
            if 'eval_env' in locals() and eval_env is not None:
                eval_env.close()
                print("Evaluation environment closed.")
        except Exception as ce:
            print(f"Warning: error closing eval_env: {ce}")
    # Close training env after finally so model.learn exceptions still handled
    try:
        env.close()
        print("Training environments closed.")
    except Exception as ce:
        print(f"Warning: error closing training env: {ce}")

    print("Training script finished.")
    # Ensure full exit (optional but helps if subprocesses linger)
    # sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', type=str, default=None, help='Continue training from this model (keeps timesteps)')
    parser.add_argument('--init-from', type=str, default=None, help='Use this pretrained model weights as initialization (timesteps reset)')
    parser.add_argument('--target-total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='Final global timesteps to reach (resume) or total to run (fresh/init)')
    parser.add_argument('--lr-initial', type=float, default=3e-4, help='Initial LR for schedule')
    parser.add_argument('--lr-final', type=float, default=5e-5, help='Final LR for schedule')
    args = parser.parse_args()
    train_agent(resume_from=args.resume_from,
                init_from=args.init_from,
                target_total_timesteps=args.target_total_timesteps,
                lr_initial=args.lr_initial,
                lr_final=args.lr_final)
