import torch


class TrainEvalResetMixin:
    """Restricts episodic metric logging to the training environments.

    Train envs are the first ``int(train_env_ratio * num_envs)`` indices. Eval envs are reset in a separate
    pass first, so the train pass overwrites ``extras["log"]``. Mix in before ``ManagerBasedRLEnv``.
    """

    # provided by the host ManagerBasedRLEnv subclass
    train_env_ratio: float

    def _reset_idx(self, env_ids):
        num_train = int(getattr(self, "train_env_ratio", 1.0) * self.num_envs)

        # env_ids may be a tensor, a sequence, or a slice, so normalize to an index tensor
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device)
        eval_ids = ids[ids >= num_train]

        # No eval envs in this batch, reset as usual.
        if eval_ids.numel() == 0:
            return super()._reset_idx(env_ids)

        train_ids = ids[ids < num_train]
        # Reset eval envs first; their logs are overwritten (or cleared) by the train pass below.
        super()._reset_idx(eval_ids)
        if train_ids.numel() > 0:
            # Repopulates extras["log"] with means over the training envs only.
            super()._reset_idx(train_ids)
        else:
            # Only eval envs reset this step, so log nothing rather than eval-only means.
            self.extras["log"] = {}
