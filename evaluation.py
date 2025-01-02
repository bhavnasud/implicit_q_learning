from typing import Dict

import flax.linen as nn
import gymnasium as gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Wrapper,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        observation, _ = observation

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
