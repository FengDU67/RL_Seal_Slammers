#!/usr/bin/env python3
"""
Debug script to identify training issues
"""

import sys
import traceback
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import MaskablePPO
from rl_seal_slammers.envs.seal_slammers_env import SealSlammersEnv

def test_env_creation():
    """Test basic environment creation"""
    print("Testing environment creation...")
    try:
        env = SealSlammersEnv()
        obs, info = env.reset()  # reset返回(obs, info)元组
        print(f"✓ Environment created successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        action_masks = env.action_masks()
        print(f"  Action mask type: {type(action_masks)}")
        print(f"  Action mask shape: {action_masks.shape}")
        print(f"  Number of valid actions: {np.sum(action_masks)}")
        return True
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        traceback.print_exc()
        return False

def test_dummy_vec_env():
    """Test DummyVecEnv wrapper"""
    print("\nTesting DummyVecEnv...")
    try:
        env = DummyVecEnv([lambda: SealSlammersEnv()])
        obs = env.reset()
        print(f"✓ DummyVecEnv created successfully")
        print(f"  Observation shape: {obs.shape}")
        
        # Test action masking
        action_masks = env.env_method("action_masks")[0]
        print(f"  Action mask type: {type(action_masks)}")
        print(f"  Action mask shape: {action_masks.shape}")
        
        # 对于扁平化的动作掩码，我们需要找到有效的动作索引
        valid_action_indices = np.where(action_masks)[0]
        if len(valid_action_indices) > 0:
            # 从扁平化索引转换回MultiDiscrete动作
            flat_action_idx = valid_action_indices[0]
            
            # 解码扁平化索引为 [object_idx, angle_idx, strength_idx]
            nvec = [3, 36, 3]  # 我们的动作空间
            strength_idx = flat_action_idx % nvec[2]
            angle_idx = (flat_action_idx // nvec[2]) % nvec[1]
            object_idx = flat_action_idx // (nvec[1] * nvec[2])
            
            action = [[object_idx, angle_idx, strength_idx]]
            obs, rewards, dones, infos = env.step(action)
            print(f"✓ Step completed successfully with action: {action[0]}")
        else:
            print("✗ No valid actions available")
            return False
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ DummyVecEnv failed: {e}")
        traceback.print_exc()
        return False

def test_subproc_vec_env():
    """Test SubprocVecEnv wrapper"""
    print("\nTesting SubprocVecEnv...")
    try:
        def make_env():
            return SealSlammersEnv()
        
        env = SubprocVecEnv([make_env for _ in range(2)])
        obs = env.reset()
        print(f"✓ SubprocVecEnv created successfully")
        print(f"  Observation shape: {obs.shape}")
        
        # Test action masking
        action_masks = env.env_method("action_masks")
        print(f"  Action masks obtained from {len(action_masks)} environments")
        print(f"  First env action mask type: {type(action_masks[0])}")
        print(f"  First env action mask shape: {action_masks[0].shape}")
        
        # Test a step with valid actions for both environments
        valid_action_indices_env1 = np.where(action_masks[0])[0]
        valid_action_indices_env2 = np.where(action_masks[1])[0]
        
        if len(valid_action_indices_env1) > 0 and len(valid_action_indices_env2) > 0:
            # 解码两个环境的第一个有效动作
            nvec = [3, 36, 3]
            
            # 环境1的动作
            flat_idx1 = valid_action_indices_env1[0]
            strength_idx1 = flat_idx1 % nvec[2]
            angle_idx1 = (flat_idx1 // nvec[2]) % nvec[1]
            object_idx1 = flat_idx1 // (nvec[1] * nvec[2])
            
            # 环境2的动作
            flat_idx2 = valid_action_indices_env2[0]
            strength_idx2 = flat_idx2 % nvec[2]
            angle_idx2 = (flat_idx2 // nvec[2]) % nvec[1]
            object_idx2 = flat_idx2 // (nvec[1] * nvec[2])
            
            actions = [[object_idx1, angle_idx1, strength_idx1],
                      [object_idx2, angle_idx2, strength_idx2]]
            obs, rewards, dones, infos = env.step(actions)
            print(f"✓ Step completed successfully with actions: {actions}")
        else:
            print("✗ No valid actions available")
            return False
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ SubprocVecEnv failed: {e}")
        traceback.print_exc()
        return False

def test_maskable_ppo():
    """Test MaskablePPO with environment"""
    print("\nTesting MaskablePPO...")
    try:
        # Use DummyVecEnv first
        env = DummyVecEnv([lambda: SealSlammersEnv()])
        
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            verbose=1,
            tensorboard_log="./debug_logs/"
        )
        print(f"✓ MaskablePPO model created successfully")
        
        # Try a very short training
        print("  Testing short training (100 steps)...")
        model.learn(total_timesteps=100)
        print(f"✓ Short training completed")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ MaskablePPO failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("=== Environment Debug Script ===\n")
    
    results = {
        "env_creation": test_env_creation(),
        "dummy_vec_env": test_dummy_vec_env(),
        "subproc_vec_env": test_subproc_vec_env(),
        "maskable_ppo": test_maskable_ppo()
    }
    
    print(f"\n=== Results ===")
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print(f"\n✓ All tests passed! The environment should work correctly.")
    else:
        print(f"\n✗ Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
