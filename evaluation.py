from typing import Dict

import flax.linen as nn
import gymnasium as gym
import numpy as np
import imageio
from tqdm import tqdm


def normalize_obs(obs, obs_min_val, obs_max_val):
    assert obs_min_val is not None and obs_max_val is not None
    normalized_obs = ((obs - obs_min_val)/(obs_max_val - obs_min_val)) * 2 - 1
    return normalized_obs

def normalize_action(action, action_min_val, action_max_val):
    assert action_min_val is not None and action_max_val is not None
    normalized_action = ((action - action_min_val)/(action_max_val - action_min_val)) * 2 - 1
    return normalized_action

def unnormalize_obs(normalized_obs, obs_min_val, obs_max_val):
    assert obs_min_val is not None and obs_max_val is not None
    obs = ((normalized_obs + 1)/2) * (obs_max_val - obs_min_val) + obs_min_val
    return obs

def unnormalize_action(normalized_action, action_min_val, action_max_val):
    assert action_min_val is not None and action_max_val is not None
    action = ((normalized_action + 1)/2) * (action_max_val - action_min_val) + action_min_val
    return action

def evaluate(agent: nn.Module, env: gym.vector.AsyncVectorEnv,
             num_episodes: int, videos_dir,
             obs_min, obs_max, actions_min, actions_max) -> Dict[str, float]:
    stats = {'return': [], 'length': [], 'success': []}

    num_envs = env.num_envs
    completed_episodes = 0
    returns = np.zeros(num_envs)
    lengths = np.zeros(num_envs)
    successes = np.zeros(num_envs)
    done_envs = np.zeros(num_envs, dtype=bool)

    obs = env.reset()[0]
    # frames = [env.render(index=0)]  # To store frames for video saving
    video_path = ""

    total_steps = 1000
    step = 0
    with tqdm(total=total_steps, desc="Evaluating steps", unit="step") as pbar:
        while completed_episodes < num_episodes:
            # Normalize observations
            obs = normalize_obs(obs, obs_min, obs_max)

            # Sample and unnormalize actions
            actions = agent.sample_actions(obs, temperature=0.0)
            actions = unnormalize_action(actions, actions_min, actions_max)

            # Step environment
            next_obs, _, terminateds, truncateds, infos = env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i, done in enumerate(dones):
                if done and not done_envs[i]:
                    done_envs[i] = True
                    completed_episodes += 1
                    returns[i] += infos['final_info'][i]['episode']['return']
                    lengths[i] += infos['final_info'][i]['episode']['length']
                    successes[i] += int(infos['final_info'][i]['episode']['success'])

                    # Save frames only for the first episode that finishes
                    # if i == 0:
                    #     frames.append(env.render(index=0))  # Save frame for the finished episode
            obs = next_obs
            done_envs = dones
            pbar.update(1)
            step += 1

    # Compute mean stats
    stats['return'] = np.mean(returns)
    stats['length'] = np.mean(lengths)
    stats['success'] = np.mean(successes)

    # Save video for the first episode only
    # if frames:
    #     videos_dir.mkdir(parents=True, exist_ok=True)
    #     video_path = videos_dir / "eval_episode_0.mp4"
    #     imageio.mimsave(video_path, frames, fps=30)

    # return stats, str(video_path)
    return stats, None
