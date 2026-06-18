from __future__ import annotations

from torch import nn


class RelaxationModel(nn.Module):
    """
    Placeholder for a future relaxation model implemented from scratch. This intented as 
    main Harmony's experiments were without relaxation
    """

    def forward(self, data):
        raise NotImplementedError("Relaxation model is intentionally empty.")
