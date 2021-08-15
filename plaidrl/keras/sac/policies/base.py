import abc
import logging

import numpy as np

# from plaidrl.torch.core import elem_or_tuple_to_numpy, torch_ify
# from plaidrl.torch.distributions import (
#     Delta,
#     GaussianMixture,
#     GaussianMixtureFull,
#     MultivariateDiagonalNormal,
#     TanhNormal,
# )
from plaidrl.keras.distributions import Delta

# import plaidrl.torch.pytorch_util as ptu
from plaidrl.policies.base import ExplorationPolicy

# from plaidrl.torch.networks import CNN, Mlp
# from plaidrl.torch.networks.basic import MultiInputSequential

# import torch
# import torch.nn.functional as F
# from torch import nn as nn


class KerasStochasticPolicy(ExplorationPolicy, metaclass=abc.ABCMeta):
    def get_action(
        self,
        obs_np,
    ):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(
        self,
        obs_np,
    ):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        return actions

    def _get_dist_from_np(self, *args, **kwargs):
        dist = self(*args, **kwargs)
        return dist


# class PolicyFromDistributionGenerator(
#     MultiInputSequential,
#     TorchStochasticPolicy,
# ):
#     """
#     Usage:
#     ```
#     distribution_generator = FancyGenerativeModel()
#     policy = PolicyFromBatchDistributionModule(distribution_generator)
#     ```
#     """

#     pass


# class MakeDeterministic(KerasStochasticPolicy):
#     def __init__(
#         self,
#         action_distribution_generator,
#     ):
#         super().__init__()
#         self._action_distribution_generator = action_distribution_generator

#     def forward(self, *args, **kwargs):
#         dist = self._action_distribution_generator.forward(*args, **kwargs)
#         return Delta(dist.mle_estimate())
