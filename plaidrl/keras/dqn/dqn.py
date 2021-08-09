import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


from collections import OrderedDict

import numpy as np

import plaidrl.keras.keras_util as kutil
from plaidrl.core.eval_util import create_stats_ordered_dict
from plaidrl.keras.keras_rl_algorithm import KerasTrainer


import keras.optimizers
import keras.losses
import keras.backend as K
from keras import Model, Input
from keras.layers import Multiply
from keras.callbacks import History


class DQNTrainer(KerasTrainer):
    def __init__(
        self,
        qf,
        target_qf,
        learning_rate=1e-3,
        soft_target_tau=1e-3,
        target_update_period=1,
        qf_criterion=None,
        discount=0.99,
        reward_scale=1.0,
    ):
        super().__init__()
        self.qf = qf
        self.target_qf = target_qf

        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.qf_optimizer = keras.optimizers.Adam(
            lr=self.learning_rate,
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or keras.losses.mse()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        # generate DQN loss update path:
        action_input = Input(shape=(self.qf.output.shape.dims[-1],))
        y_pred = Multiply()([self.qf.output, action_input])
        y_pred = kutil.Sum(axis=1, keepdims=True)(y_pred)

        self.qf_pred = Model(inputs=[self.qf.input, action_input], outputs=y_pred)
        self.qf_pred.compile(loss=self.qf_criterion, optimizer=self.qf_optimizer)

    def train_from_keras(self, batch):
        rewards = batch["rewards"] * self.reward_scale
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        Compute loss
        """
        target_q_values = self.target_qf.predict(next_obs).max(1, keepdims=True)
        y_target = rewards + (1.0 - terminals) * self.discount * target_q_values

        # actions is a one-hot vector
        # y_pred = np.sum(self.qf.predict(obs) * actions, axis=1, keepdims=True)

        # replicate the y_pred
        result = self.qf_pred.fit([obs, actions], y_target, verbose=0)
        y_pred = self.qf_pred.predict([obs, actions])

        """
        Soft target network updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            kutil.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics["QF Loss"] = np.mean(result.history["loss"])
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Y Predictions",
                    y_pred,
                )
            )
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.qf,
            self.target_qf,
        ]

    def get_snapshot(self):
        return dict(
            qf=self.qf,
            target_qf=self.target_qf,
        )
