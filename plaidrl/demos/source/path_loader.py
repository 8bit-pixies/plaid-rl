import glob
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn as nn

import plaidrl.torch.pytorch_util as ptu
from plaidrl.core import logger
from plaidrl.core.eval_util import create_stats_ordered_dict
from plaidrl.data_management.path_builder import PathBuilder
from plaidrl.torch.core import np_to_pytorch_batch
from plaidrl.torch.torch_rl_algorithm import TorchTrainer
from plaidrl.util.io import load_local_or_remote_file

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


class PathLoader:
    """
    Loads demonstrations and/or off-policy data into a Trainer
    """

    def load_demos(
        self,
    ):
        pass
