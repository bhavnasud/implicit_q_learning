import collections
from typing import Optional

import d4rl
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import lerobot
from pathlib import Path
from hydra import compose, initialize_config_dir
from lerobot.common.datasets.factory import make_dataset
import torch

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

class D3ILDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 task: str):
        # lerobot_root = Path(lerobot.__path__[0])
        # config_path = lerobot_root / "configs"

        # with initialize_config_dir(config_dir=str(config_path)):
        #     if task == 'sorting':
        #         hydra_cfg = compose(
        #             config_name="default.yaml", overrides=["policy=vqbet_d3il_sorting", "env=d3il_sorting_state"]
        #         )
        #     elif task == 'stacking':
        #         hydra_cfg = compose(
        #             config_name="default.yaml", overrides=["policy=vqbet_d3il_stacking", "env=d3il_stacking_state"]
        #         )

        # trainset = make_dataset(hydra_cfg, root=hydra_cfg.dataset_root)
        # print(trainset.hf_dataset)

        # if task == 'stacking':
        #     trainset_observations = np.concatenate(
        #         (
        #             np.array(trainset.hf_dataset['observation.state']),
        #             np.array(trainset.hf_dataset['observation.environment_state']),
        #         ),
        #         axis=-1
        #     )
        # else:
        #     trainset_observations = np.array(trainset.hf_dataset['observation.state'])

        # next_observations = np.zeros_like(trainset_observations)
        # done_mask = np.array(trainset.hf_dataset['next.done'])
        # next_observations[:-1] = trainset_observations[1:]

        # # TODO: get true next_observations from pkl files in this case
        # next_observations[done_mask] = trainset_observations[done_mask]

        # super().__init__(
        #     observations=trainset_observations.astype(np.float32),
        #     actions=np.array(trainset.hf_dataset['action'], np.float32),
        #     rewards=np.array(trainset.hf_dataset['next.reward'], np.float32),
        #     masks=np.zeros_like(trainset.hf_dataset['next.reward']),
        #     dones_float=done_mask.astype(np.float32),
        #     next_observations=next_observations.astype(np.float32),
        #     size=len(trainset.hf_dataset['observation.state'])
        # )

        dataset = None
        assert task in ['sorting', 'stacking', 'stacking-action-noise', 'sorting-suboptimal-demos', 'pusht', 'pusht-action-noise', 'pusht-suboptimal-demos']
        if task == 'sorting':
            dataset = torch.load("/home/bsud/data/d3il_sorting.pt")
        elif task == 'stacking':
            dataset = torch.load("/home/bsud/data/d3il_stacking.pt")
        elif task == 'stacking-action-noise':
            dataset = torch.load("/home/bsud/data/d3il_stacking_action_noise.pt")
        elif task == 'sorting-suboptimal-demos':
            dataset = torch.load("/home/bsud/data/d3il_sorting_suboptimal_demos.pt")
        elif task == 'pusht':
            dataset = torch.load("/home/bsud/data/pusht.pt")
        elif task == 'pusht-action-noise':
            dataset = torch.load("/home/bsud/data/pusht-action-noise.pt")
        elif task == 'pusht-suboptimal-demos':
            dataset = torch.load("/home/bsud/data/pusht-suboptimal-demos.pt")
        
        dataset = dataset["fields"]
        observations = np.array(dataset["obs"])
        next_observations = np.zeros_like(observations)
        done_mask = np.array(dataset["done"])
        next_observations[:-1] = observations[1:]
        # TODO: get true next_observations from pkl files in this case
        next_observations[done_mask] = observations[done_mask]
        super().__init__(
            observations=observations.astype(np.float32),
            actions=np.array(dataset["action"], np.float32),
            rewards=np.array(dataset["reward"], np.float32),
            masks=np.zeros_like(dataset["reward"]),
            dones_float=done_mask.astype(np.float32),
            next_observations=next_observations.astype(np.float32),
            size=len(observations)
        )

class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
