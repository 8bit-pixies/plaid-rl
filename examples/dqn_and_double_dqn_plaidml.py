"""
Run DQN on CartPole-v0.
Using Plaidml instead of pytorch
"""

import gym
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


from plaidrl.data_management.env_replay_buffer import EnvReplayBuffer
from plaidrl.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from plaidrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from plaidrl.launchers.launcher_util import setup_logger
from plaidrl.policies.keras_argmax import ArgmaxDiscretePolicy
from plaidrl.samplers.data_collector import MdpPathCollector
from plaidrl.keras.dqn.dqn import DQNTrainer
from plaidrl.keras.dqn.double_dqn import DoubleDQNTrainer
from plaidrl.keras.networks import mlp_builder
from plaidrl.keras.keras_rl_algorithm import KerasBatchRLAlgorithm

import keras.losses
import keras.optimizers


def experiment(variant):
    expl_env = gym.make("CartPole-v0").env
    eval_env = gym.make("CartPole-v0").env
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    qf = mlp_builder(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
    target_qf = mlp_builder(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
    qf_criterion = keras.losses.mse
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    # DoubleDQNTrainer
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant["trainer_kwargs"]
    )
    replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
    )
    algorithm = KerasBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"]
    )
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="DQN",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1e6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3e-4,
        ),
    )
    setup_logger("dqn-CartPole", variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
