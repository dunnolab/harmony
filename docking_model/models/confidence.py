from __future__ import annotations

from torch import nn


class ConfidenceModel(nn.Module):
    """Placeholder for a standalone confidence model

    The score model already has confidence/affinity heads. This module exists so a 
    separate confidence module could be integrated to Harmony as well
    """

    def forward(self, data):
        raise NotImplementedError("Standalone confidence model is intentionally empty.")
