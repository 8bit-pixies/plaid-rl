import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import plaidrl.torch.pytorch_util as ptu
from plaidrl.policies.base import ExplorationPolicy
from plaidrl.torch.core import elem_or_tuple_to_numpy, torch_ify
from plaidrl.torch.distributions import (
    Delta,
    GaussianMixture,
    GaussianMixtureFull,
    MultivariateDiagonalNormal,
    TanhNormal,
)
from plaidrl.torch.networks import CNN, Mlp
from plaidrl.torch.networks.basic import MultiInputSequential
from plaidrl.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator,
)
from plaidrl.torch.sac.policies.base import (
    MakeDeterministic,
    PolicyFromDistributionGenerator,
    TorchStochasticPolicy,
)


class LatentVariableModel(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
