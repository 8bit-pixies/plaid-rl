"""
Torch argmax policy
"""
import numpy as np
from torch import nn

# import plaidrl.torch.pytorch_util as ptu
from plaidrl.policies.base import Policy


class ArgmaxDiscretePolicy(Policy):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        q_values = self.qf.predict(obs)
        return q_values.argmax(), {}
