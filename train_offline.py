import os
from typing import Tuple
import gymnasium as gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import D4RLDataset, D3ILDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner
import wandb
from pathlib import Path
import gym_sorting.envs
import gym_stacking.envs

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'gym_sorting/sorting-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 50,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 10000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e8), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_string('task', 'sorting', 'task name')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0

def normalize_obs(obs, obs_min_val, obs_max_val):
    assert obs_min_val is not None and obs_max_val is not None
    normalized_obs = ((obs - obs_min_val)/(obs_max_val - obs_min_val)) * 2 - 1
    return normalized_obs

def normalize_action(self, action, action_min_val, action_max_val):
    assert action_min_val is not None and action_max_val is not None
    normalized_action = ((action - action_min_val)/(action_max_val - action_min_val)) * 2 - 1
    return normalized_action

def unnormalize_obs(self, normalized_obs, obs_min_val, obs_max_val):
    assert obs_min_val is not None and obs_max_val is not None
    obs = ((normalized_obs + 1)/2) * (obs_max_val - obs_min_val) + obs_min_val
    return obs

def unnormalize_action(self, normalized_action, action_min_val, action_max_val):
    assert action_min_val is not None and action_max_val is not None
    action = ((normalized_action + 1)/2) * (action_max_val - action_min_val) + action_min_val
    return action

def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    kwargs = {
		"max_steps_per_episode": 1000,
		"if_vision": False,
		"render": False,
		"self_start": True
	}
    env = gym.make(FLAGS.env_name, disable_env_checker=True, **kwargs)

    env = wrappers.D3ILEnvWrapper(env, FLAGS.task)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # dataset = D4RLDataset(env)
    dataset = D3ILDataset(env)
    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)
    elif 'sort' in FLAGS.task or 'stack' in FLAGS.task:
        dataset.rewards *= 1000.0
        obs_min = np.min(dataset.observations, axis=0)
        obs_max = np.max(dataset.observations, axis=0)
        actions_min = np.min(dataset.actions, axis=0)
        actions_max = np.max(dataset.actions, axis=0)
        dataset.observations = normalize_obs(dataset.observations, obs_min, obs_max)
        dataset.next_observations = normalize_obs(dataset.next_observations, obs_min, obs_max)
        dataset.actions = normalize_action(dataset.actions, actions_min, actions_max)
        return env, dataset, obs_min, obs_max, actions_min, actions_max

    return env, dataset

import flax.serialization
import jax.numpy as jnp
import os

def save_checkpoint(learner: Learner, checkpoint_dir: str, step: int):
    """
    Saves a checkpoint of the Learner model.

    Args:
        learner: An instance of the Learner class.
        checkpoint_dir: Directory to save the checkpoint files.
        step: Current training step, used to differentiate checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.npz")

    checkpoint_data = {
        "rng": learner.rng,
        "actor": flax.serialization.to_state_dict(learner.actor),
        "critic": flax.serialization.to_state_dict(learner.critic),
        "value": flax.serialization.to_state_dict(learner.value),
        "target_critic": flax.serialization.to_state_dict(learner.target_critic),
    }

    with open(checkpoint_path, "wb") as f:
        f.write(flax.serialization.to_bytes(checkpoint_data))
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(learner: Learner, checkpoint_path: str):
    """
    Loads a checkpoint into the Learner model.

    Args:
        learner: An instance of the Learner class.
        checkpoint_path: Path to the checkpoint file.
    """
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = flax.serialization.from_bytes({}, f.read())

    learner.rng = checkpoint_data["rng"]
    learner.actor = learner.actor.replace(
        **flax.serialization.from_state_dict(learner.actor, checkpoint_data["actor"])
    )
    learner.critic = learner.critic.replace(
        **flax.serialization.from_state_dict(learner.critic, checkpoint_data["critic"])
    )
    learner.value = learner.value.replace(
        **flax.serialization.from_state_dict(learner.value, checkpoint_data["value"])
    )
    learner.target_critic = learner.target_critic.replace(
        **flax.serialization.from_state_dict(learner.target_critic, checkpoint_data["target_critic"])
    )
    print(f"Checkpoint loaded from {checkpoint_path}")


def main(_):
    wandb.init(
			project="iql",
			entity="dmanip-rss",
			name=f"iql_{FLAGS.task}_{FLAGS.seed}",
			group=FLAGS.task,
			tags=[f"task:{FLAGS.task}", f"seed:{FLAGS.seed}"],
			dir="/juno/u/bsud2/iql_logs",
            config=FLAGS.flag_values_dict()
		)
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                                str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset, obs_min, obs_max, actions_min, actions_max = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    eval_returns = []
    best_return = 0
    videos_dir = Path(os.path.join(FLAGS.save_dir, 'videos', FLAGS.task, str(FLAGS.seed)))
    last_checkpoint_dir = os.path.join(FLAGS.save_dir, 'checkpoints', FLAGS.task, str(FLAGS.seed), 'last')
    best_checkpoint_dir = os.path.join(FLAGS.save_dir, 'checkpoints', FLAGS.task, str(FLAGS.seed), 'best')
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                    wandb.log({f'training/{k}': v}, step=i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
                    wandb.log({f'training/{k}_hist': wandb.Histogram(v)}, step=i)
            summary_writer.flush()
            save_checkpoint(agent, last_checkpoint_dir, i)
  
        if i % FLAGS.eval_interval == 0:
            eval_stats, video_path = evaluate(agent, env, FLAGS.eval_episodes, videos_dir, obs_min, obs_max, actions_min, actions_max)
            wandb_video = wandb.Video(video_path, fps=30, format="mp4")
            wandb.log({"eval/video": wandb_video}, step=i)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
                wandb.log({f'evaluation/average_{k}s': v}, step=i)
                print("logged to wandb")
            if eval_stats['return'] > best_return:
                best_return = eval_stats['return']
                save_checkpoint(agent, best_checkpoint_dir, i)

            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
