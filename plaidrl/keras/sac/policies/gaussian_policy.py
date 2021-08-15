import abc
import logging
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


import keras
import numpy as np
import keras.backend as K

# from plaidrl.torch.core import elem_or_tuple_to_numpy, torch_ify
# from plaidrl.torch.distributions import (
#     Delta,
#     GaussianMixture,
#     GaussianMixtureFull,
#     MultivariateDiagonalNormal,
#     TanhNormal,
# )
# from plaidrl.torch.networks import CNN, Mlp
# from plaidrl.torch.networks.basic import MultiInputSequential
# from plaidrl.torch.networks.stochastic.distribution_generator import (
#     DistributionGenerator,
# )
from plaidrl.keras.sac.policies.base import KerasStochasticPolicy

# import plaidrl.torch.pytorch_util as ptu
from plaidrl.keras.networks import mlp_builder
from keras import Input, Model, activations
from keras.layers import Concatenate, Dense, Layer, Lambda
import keras.backend as K
from plaidrl.policies.base import ExplorationPolicy, Policy


# import torch
# import torch.nn.functional as F
# from torch import nn as nn


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# TODO: deprecate classes below in favor for PolicyFromDistributionModule


# class TanhGaussianPolicyAdapter(TorchStochasticPolicy):
#     """
#     Usage:

#     ```
#     obs_processor = ...
#     policy = TanhGaussianPolicyAdapter(obs_processor)
#     ```
#     """

#     def __init__(
#         self,
#         obs_processor,
#         obs_processor_output_dim,
#         action_dim,
#         hidden_sizes,
#     ):
#         super().__init__()
#         self.obs_processor = obs_processor
#         self.obs_processor_output_dim = obs_processor_output_dim
#         self.mean_and_log_std_net = Mlp(
#             hidden_sizes=hidden_sizes,
#             output_size=action_dim * 2,
#             input_size=obs_processor_output_dim,
#         )
#         self.action_dim = action_dim

#     def forward(self, obs):
#         h = self.obs_processor(obs)
#         h = self.mean_and_log_std_net(h)
#         mean, log_std = torch.split(h, self.action_dim, dim=1)
#         log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
#         std = torch.exp(log_std)

#         tanh_normal = TanhNormal(mean, std)
#         return tanh_normal


def create_gaussian_network(hidden_sizes, obs_dim, action_dim):
    inputs = Input(shape=(obs_dim,))
    for idx, hidden_size in enumerate(hidden_sizes):
        if idx == 0:
            x = Dense(hidden_size, activation=activations.relu)(inputs)
        else:
            x = Dense(hidden_size, activation=activations.relu)(x)

    mean = Dense(action_dim)(x)
    log_std = Dense(action_dim)(x)
    log_std = Lambda(lambda x: K.clip(x, LOG_SIG_MIN, LOG_SIG_MAX))(log_std)
    std = Lambda(lambda x: K.exp(x))(log_std)

    # now TanhNormal(std) is a dist which can sample from
    # and have a wrapper for stochastic/discrete
    # get action object
    return Model(inputs=inputs, outputs=[mean, std])


class Normal(object):
    """
    Trying to copy this interface:
    https://github.com/tensorflow/probability/blob/v0.13.0/tensorflow_probability/python/distributions/normal.py#L45-L254
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def log_prob(self, x):
        log_unnormalized = (
            -0.5
            * ((x / self.std) - (self.self.mean / self.std))
            * ((x / self.std) - (self.self.mean / self.std))
        )
        log_normalization = K.constant(0.5 * np.log(2.0 * np.pi)) + K.log(self.std)
        return log_unnormalized - log_normalization


class TanhGaussianPolicy(Policy):
    def __init__(self, gaussian_network, is_deterministic=False):
        self.gaussian_network = gaussian_network
        self.is_deterministic = is_deterministic

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        mean, std = self.gaussian_network.predict(obs)

        if self.is_deterministic:
            # print(mean, std)
            return np.squeeze(np.tanh(mean), 0), {}

        # this would be someting like:
        # actions = dist(mean, std).sample()
        z = mean + std * np.random.normal(mean, std)

        # print(mean, std, z)
        return np.squeeze(np.tanh(z), 0), {}
