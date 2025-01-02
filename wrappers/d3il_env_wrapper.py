import gymnasium as gym
import gym_sorting
import numpy as np
from gymnasium.spaces import Dict, Box

class D3ILEnvWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.env = env
		obs_shape = env.observation_space["agent_pos"].shape
		self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs_shape)

	def reset(self):
		state = self.env.reset()[0]["agent_pos"]
		return state, {}

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.copy())
		state = obs["agent_pos"]
		info['success'] = info['is_success']
		return state, reward, terminated, truncated, info

	def render(self, *args, **kwargs):
		return self.env.render()