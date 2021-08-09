import abc
from collections import OrderedDict
from typing import Iterable

from torch import nn as nn

from plaidrl.core.batch_rl_algorithm import BatchRLAlgorithm
from plaidrl.core.online_rl_algorithm import OnlineRLAlgorithm
from plaidrl.core.trainer import Trainer

# from plaidrl.keras.core import np_to_pytorch_batch


class KerasOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode=None):
        pass


class KerasBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode=None):
        pass


class KerasTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch):
        self._num_train_steps += 1
        self.train_from_keras(np_batch)

    def get_diagnostics(self):
        return OrderedDict(
            [
                ("num train calls", self._num_train_steps),
            ]
        )

    @abc.abstractmethod
    def train_from_keras(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass
