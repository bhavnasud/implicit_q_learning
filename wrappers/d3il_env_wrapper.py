import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Box

class D3ILEnvWrapper(gym.Wrapper):
	def __init__(self, env, task):
		super().__init__(env)
		self.env = env
		self.task = task
		state_shape = env.observation_space["agent_pos"].shape
		env_state_shape = env.observation_space["environment_state"].shape
		concatenated_shape = (state_shape[0] + env_state_shape[0],)
		if task == 'sorting':
			self.observation_space = Box(low=-np.inf, high=np.inf, shape=state_shape)
		else:
			self.observation_space = Box(low=-np.inf, high=np.inf, shape=concatenated_shape)

	def reset(self):
		obs = self.env.reset()[0]
		agent_pos = obs["agent_pos"]
		environment_state = obs["environment_state"]
		if self.task == 'sorting':
			return agent_pos, {}
		else:
			return np.concatenate([agent_pos, environment_state]), {}

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.copy())
		agent_pos = obs["agent_pos"]
		environment_state = obs["environment_state"]
		if self.task == 'sorting':
			state = agent_pos
		else:
			state = np.concatenate([agent_pos, environment_state])
		info['success'] = info['is_success']
		return state, reward, terminated, truncated, info

	def render(self, *args, **kwargs):
		return self.env.render()