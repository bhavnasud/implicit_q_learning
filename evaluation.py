from typing import Dict

import flax.linen as nn
import gymnasium as gym
import numpy as np
import imageio
from train_offline import normalize_obs, unnormalize_action

def evaluate(agent: nn.Module, env: gym.Wrapper,
             num_episodes: int, videos_dir,
             obs_min, obs_max, actions_min, actions_max) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    video_path = ""
    frames = []
    for i in range(num_episodes):
        observation, done = env.reset(), False
        if i == 0:
            frames.append(env.render())
        observation, _ = observation

        while not done:
            observation = normalize_obs(observation, obs_min, obs_max)
            action = agent.sample_actions(observation, temperature=0.0)
            action = unnormalize_action(action, actions_min, actions_max)
            observation, _, terminated, truncated, info = env.step(action)
            if i == 0:
                frames.append(env.render())
            done = terminated or truncated

        for k in stats.keys():
            stats[k].append(info['episode'][k])
        if i == 0:
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_path = videos_dir / f"eval_episode_0.mp4"
            imageio.mimsave(video_path, frames, fps=30)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, str(video_path)
