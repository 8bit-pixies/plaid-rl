import os

import gym

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import plaidrl.torch.pytorch_util as ptu
from plaidrl.data_management.env_replay_buffer import EnvReplayBuffer
from plaidrl.keras.keras_rl_algorithm import KerasBatchRLAlgorithm
from plaidrl.keras.networks import concat_mlp_builder
from plaidrl.keras.sac.policies import TanhGaussianPolicy, create_gaussian_network
from plaidrl.keras.sac.sac import SACTrainer
from plaidrl.launchers.launcher_util import setup_logger
from plaidrl.samplers.data_collector import MdpPathCollector


def experiment(variant):
    expl_env = gym.make("Pendulum-v0").env
    eval_env = gym.make("Pendulum-v0").env
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant["layer_size"]
    qf1 = concat_mlp_builder(
        input_size=[obs_dim, action_dim],
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = concat_mlp_builder(
        input_size=[obs_dim, action_dim],
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = concat_mlp_builder(
        input_size=[obs_dim, action_dim],
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = concat_mlp_builder(
        input_size=[obs_dim, action_dim],
        output_size=1,
        hidden_sizes=[M, M],
    )
    base_policy = create_gaussian_network(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(base_policy)
    eval_policy = TanhGaussianPolicy(base_policy, True)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant["trainer_kwargs"]
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
        algorithm="SAC",
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
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger("name-of-experiment", variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
