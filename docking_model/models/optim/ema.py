from __future__ import annotations

import torch


class ExponentialMovingAverage:
    def __init__(self, parameters, decay: float, use_num_updates: bool = True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError("EMA decay must be between 0 and 1.")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [param.detach().clone() for param in parameters if param.requires_grad]
        self.collected_params: list[torch.Tensor] = []

    def update(self, parameters) -> None:
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            trainable = [param for param in parameters if param.requires_grad]
            for shadow, param in zip(self.shadow_params, trainable):
                shadow.sub_(one_minus_decay * (shadow - param.detach()))

    def copy_to(self, parameters) -> None:
        trainable = [param for param in parameters if param.requires_grad]
        for shadow, param in zip(self.shadow_params, trainable):
            param.data.copy_(shadow.data)

    def store(self, parameters) -> None:
        self.collected_params = [param.detach().clone() for param in parameters]

    def restore(self, parameters) -> None:
        for collected, param in zip(self.collected_params, parameters):
            param.data.copy_(collected.data)
        self.collected_params = []

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict: dict, device) -> None:
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = [tensor.to(device) for tensor in state_dict["shadow_params"]]
