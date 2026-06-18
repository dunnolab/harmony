from __future__ import annotations

from typing import Any

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import beta


def bridge_transform_t(t, alpha: float):
    return (np.exp(alpha * t) - np.exp(alpha)) / (1 - np.exp(alpha))


def sinusoidal_embedding(timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 10000) -> torch.Tensor:
    if timesteps.dim() != 1:
        raise ValueError("sinusoidal_embedding expects a one-dimensional timestep tensor.")
    half_dim = embedding_dim // 2
    scale = math.log(max_positions) / (half_dim - 1)
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -scale
    )
    emb = timesteps.float()[:, None] * frequencies[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    return emb


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def get_timestep_embedding(embedding_type: str, embedding_dim: int, embedding_scale: float = 10000):
    if embedding_type == "sinusoidal":
        return lambda x: sinusoidal_embedding(embedding_scale * x, embedding_dim)
    if embedding_type == "fourier":
        return GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    raise ValueError(f"Unsupported timestep embedding type: {embedding_type}")


def t_to_sigma(t_dict: dict[str, Any], sigma_cfg) -> dict[str, Any]:
    tr_sigma = exp_sigma(t_dict["tr"], sigma_cfg.tr_sigma_min, sigma_cfg.tr_sigma_max)
    rot_sigma = exp_sigma(t_dict["rot"], sigma_cfg.rot_sigma_min, sigma_cfg.rot_sigma_max)
    tor_sigma = exp_sigma(t_dict["tor"], sigma_cfg.tor_sigma_min, sigma_cfg.tor_sigma_max)

    sc_tor_sigma = None
    if "sc_tor" in t_dict and t_dict["sc_tor"] is not None:
        sidechain_tor_transform_type = getattr(sigma_cfg, "sidechain_tor_transform_type", None)
        if sidechain_tor_transform_type is not None:
            sidechain_tor_bridge = str(sidechain_tor_transform_type).lower() == "bridge"
        else:
            sidechain_tor_bridge = getattr(sigma_cfg, "sidechain_tor_bridge", None)

        if (
            sidechain_tor_bridge is False
            and getattr(sigma_cfg, "sidechain_tor_sigma_min", None) is not None
            and getattr(sigma_cfg, "sidechain_tor_sigma_max", None) is not None
        ):
            sc_tor_sigma = exp_sigma(
                t_dict["sc_tor"],
                sigma_cfg.sidechain_tor_sigma_min,
                sigma_cfg.sidechain_tor_sigma_max,
            )
        elif sidechain_tor_bridge is True and getattr(sigma_cfg, "sidechain_tor_sigma", None) is not None:
            sc_tor_sigma = constant_like(t_dict["sc_tor"], sigma_cfg.sidechain_tor_sigma)
        elif (
            sidechain_tor_bridge is None
            and getattr(sigma_cfg, "sidechain_tor_sigma_min", None) is not None
            and getattr(sigma_cfg, "sidechain_tor_sigma_max", None) is not None
        ):
            sc_tor_sigma = exp_sigma(
                t_dict["sc_tor"],
                sigma_cfg.sidechain_tor_sigma_min,
                sigma_cfg.sidechain_tor_sigma_max,
            )
        elif getattr(sigma_cfg, "sidechain_tor_sigma", None) is not None:
            sc_tor_sigma = constant_like(t_dict["sc_tor"], sigma_cfg.sidechain_tor_sigma)

    bb_tr_sigma = None
    bb_rot_sigma = None
    if "bb_tr" in t_dict and t_dict["bb_tr"] is not None:
        bb_tr_sigma = constant_like(t_dict["bb_tr"], sigma_cfg.bb_tr_sigma)
        bb_rot_sigma = constant_like(t_dict["bb_rot"], sigma_cfg.bb_rot_sigma)

    return {
        "tr_sigma": tr_sigma,
        "rot_sigma": rot_sigma,
        "tor_sigma": tor_sigma,
        "sc_tor_sigma": sc_tor_sigma,
        "bb_tr_sigma": bb_tr_sigma,
        "bb_rot_sigma": bb_rot_sigma,
    }


def get_schedules(
    inference_steps: int,
    sampling_alpha: float,
    sampling_beta: float,
    bb_tr_bridge_alpha: float | None = None,
    bb_rot_bridge_alpha: float | None = None,
    sc_tor_bridge_alpha: float | None = None,
    sidechain_tor_bridge: bool = True,
    t_max: float = 1.0,
) -> dict[str, np.ndarray | None]:
    t_schedule = beta_schedule(inference_steps, sampling_alpha, sampling_beta, t_max=t_max)
    schedules = {
        "t": t_schedule,
        "tr": t_schedule,
        "rot": t_schedule,
        "tor": t_schedule,
        "bb_tr": None,
        "bb_rot": None,
        "sc_tor": None,
    }

    if bb_tr_bridge_alpha is not None:
        schedules["bb_tr"] = bridge_transform_t(t_schedule, bb_tr_bridge_alpha)
    if bb_rot_bridge_alpha is not None:
        schedules["bb_rot"] = bridge_transform_t(t_schedule, bb_rot_bridge_alpha)
    if sc_tor_bridge_alpha is not None and sidechain_tor_bridge:
        schedules["sc_tor"] = bridge_transform_t(t_schedule, sc_tor_bridge_alpha)
    else:
        schedules["sc_tor"] = t_schedule
    return schedules


def set_time(
    complex_graphs,
    t: float | None,
    t_tr: float,
    t_rot: float,
    t_tor: float,
    t_sidechain_tor: float | None,
    t_bb_tr: float | None,
    t_bb_rot: float | None,
    batch_size: int,
    all_atoms: bool,
    device,
) -> None:
    t_value = t_tr if t is None else t
    complex_graphs["ligand"].node_t = node_time_dict(complex_graphs["ligand"].num_nodes, device, t_tr, t_rot, t_tor)
    complex_graphs["receptor"].node_t = node_time_dict(complex_graphs["receptor"].num_nodes, device, t_tr, t_rot, t_tor)
    complex_graphs["ligand"].node_t["t"] = t_value * torch.ones(complex_graphs["ligand"].num_nodes, device=device)
    complex_graphs["receptor"].node_t["t"] = t_value * torch.ones(complex_graphs["receptor"].num_nodes, device=device)
    complex_graphs.complex_t = {
        "tr": t_tr * torch.ones(batch_size, device=device),
        "rot": t_rot * torch.ones(batch_size, device=device),
        "tor": t_tor * torch.ones(batch_size, device=device),
        "t": t_value * torch.ones(batch_size, device=device),
    }
    if all_atoms:
        complex_graphs["atom"].node_t = node_time_dict(complex_graphs["atom"].num_nodes, device, t_tr, t_rot, t_tor)
        complex_graphs["atom"].node_t["t"] = t_value * torch.ones(complex_graphs["atom"].num_nodes, device=device)

    if t_sidechain_tor is not None:
        set_extra_time(complex_graphs, "sc_tor", t_sidechain_tor, batch_size, all_atoms, device)
    if t_bb_tr is not None:
        set_extra_time(complex_graphs, "bb_tr", t_bb_tr, batch_size, all_atoms, device)
        set_extra_time(complex_graphs, "bb_rot", t_bb_rot, batch_size, all_atoms, device)


def set_time_t_dict(
    complex_graphs,
    t_dict: dict[str, Any],
    batch_size: int,
    all_atoms: bool,
    device,
    include_miscellaneous_atoms: bool = False,
) -> None:
    del include_miscellaneous_atoms
    set_time(
        complex_graphs,
        t=t_dict.get("t"),
        t_tr=t_dict["tr"],
        t_rot=t_dict["rot"],
        t_tor=t_dict["tor"],
        t_sidechain_tor=t_dict.get("sc_tor"),
        t_bb_tr=t_dict.get("bb_tr"),
        t_bb_rot=t_dict.get("bb_rot"),
        batch_size=batch_size,
        all_atoms=all_atoms,
        device=device,
    )


def exp_sigma(t, sigma_min: float, sigma_max: float):
    return sigma_min ** (1 - t) * sigma_max**t


def constant_like(t, value: float | None):
    if value is None:
        return None
    if torch.is_tensor(t):
        return torch.ones_like(t) * value
    if isinstance(t, np.ndarray):
        return np.ones_like(t) * value
    return value


def beta_schedule(inference_steps: int, alpha: float, beta_value: float, t_max: float):
    lin_max = beta.cdf(t_max, a=alpha, b=beta_value)
    values = np.linspace(lin_max, 0, inference_steps + 1)[:-1]
    return beta.ppf(values, a=alpha, b=beta_value)


def node_time_dict(num_nodes: int, device, t_tr: float, t_rot: float, t_tor: float):
    return {
        "tr": t_tr * torch.ones(num_nodes, device=device),
        "rot": t_rot * torch.ones(num_nodes, device=device),
        "tor": t_tor * torch.ones(num_nodes, device=device),
    }


def set_extra_time(complex_graphs, key: str, value: float, batch_size: int, all_atoms: bool, device) -> None:
    complex_graphs["ligand"].node_t[key] = value * torch.ones(complex_graphs["ligand"].num_nodes, device=device)
    complex_graphs["receptor"].node_t[key] = value * torch.ones(complex_graphs["receptor"].num_nodes, device=device)
    complex_graphs.complex_t[key] = value * torch.ones(batch_size, device=device)
    if all_atoms:
        complex_graphs["atom"].node_t[key] = value * torch.ones(complex_graphs["atom"].num_nodes, device=device)
