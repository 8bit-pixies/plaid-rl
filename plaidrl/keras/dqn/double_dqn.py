import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np

import plaidrl.keras.keras_util as kutil
from plaidrl.core.eval_util import create_stats_ordered_dict
from plaidrl.keras.dqn.dqn import DQNTrainer


class DoubleDQNTrainer(DQNTrainer):
    def train_from_keras(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        Compute loss
        """
        best_action_idxs = self.qf.predict(next_obs).argmax(1)
        target_q_values = np.take_along_axis(
            self.target_qf.predict(next_obs),
            np.expand_dims(best_action_idxs, axis=1),
            axis=1,
        )
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
