from __future__ import annotations

import numpy as np
import torch


class HarmonicPrior:
    def __init__(self, bb_random_prior_ot: int = 1, bb_random_prior_std: float = 0.1, bb_random_prior_ot_inf: int = 1):
        self.bb_random_prior_ot = int(bb_random_prior_ot)
        self.bb_random_prior_std = bb_random_prior_std
        self.bb_random_prior_ot_inf = int(bb_random_prior_ot_inf)

    def sample_harmonic_noise(self, n_residues: int, n_samples: int, alpha: float = 3 / (3.8**2), std: float = 0.1):
        j_mat = torch.zeros(n_residues, n_residues)
        for i, j in zip(np.arange(n_residues - 1), np.arange(1, n_residues)):
            j_mat[i, i] += alpha
            j_mat[j, j] += alpha
            j_mat[i, j] = j_mat[j, i] = -alpha
        eigvals, eigvecs = torch.linalg.eigh(j_mat)
        eigvals_inv = 1 / eigvals
        eigvals_inv[0] = 0
        noise_shape = (n_samples, n_residues, 3) if n_samples > 1 else (n_residues, 3)
        return eigvecs @ (torch.sqrt(eigvals_inv)[:, None] * torch.randn(noise_shape)) * std

    def __call__(self, calpha_apo, calpha_holo):
        return best_prior_sample(
            calpha_apo=calpha_apo,
            calpha_holo=calpha_holo,
            n_samples=self.bb_random_prior_ot,
            std=self.bb_random_prior_std,
            sampler=self.sample_harmonic_noise,
        )

    def sample_for_inference(self, complex_graph):
        return sample_prior_for_inference(
            complex_graph=complex_graph,
            n_samples=self.bb_random_prior_ot_inf,
            std=self.bb_random_prior_std,
            sampler=self.sample_harmonic_noise,
        )


class GaussianPrior:
    def __init__(self, bb_random_prior_ot: int = 1, bb_random_prior_std: float = 0.1, bb_random_prior_ot_inf: int = 1):
        self.bb_random_prior_ot = int(bb_random_prior_ot)
        self.bb_random_prior_std = bb_random_prior_std
        self.bb_random_prior_ot_inf = int(bb_random_prior_ot_inf)

    @staticmethod
    def sample_gaussian_noise(n_residues: int, n_samples: int, std: float = 0.1):
        shape = (n_samples, n_residues, 3) if n_samples > 1 else (n_residues, 3)
        return torch.randn(shape) * std

    def __call__(self, calpha_apo, calpha_holo):
        return best_prior_sample(
            calpha_apo=calpha_apo,
            calpha_holo=calpha_holo,
            n_samples=self.bb_random_prior_ot,
            std=self.bb_random_prior_std,
            sampler=self.sample_gaussian_noise,
        )

    def sample_for_inference(self, complex_graph):
        return sample_prior_for_inference(
            complex_graph=complex_graph,
            n_samples=self.bb_random_prior_ot_inf,
            std=self.bb_random_prior_std,
            sampler=self.sample_gaussian_noise,
        )


def construct_bb_prior(args):
    if not getattr(args, "bb_random_prior", False):
        return None
    noise = getattr(args, "bb_random_prior_noise", "gaussian")
    prior_cls = GaussianPrior if noise == "gaussian" else HarmonicPrior if noise == "harmonic" else None
    if prior_cls is None:
        raise ValueError(f"Unsupported backbone prior noise type: {noise}")
    return prior_cls(
        bb_random_prior_ot=getattr(args, "bb_random_prior_ot", 1),
        bb_random_prior_std=getattr(args, "bb_random_prior_std", 0.1),
        bb_random_prior_ot_inf=getattr(args, "bb_random_prior_ot_inf", 1),
    )


def best_prior_sample(calpha_apo, calpha_holo, n_samples: int, std: float, sampler):
    if n_samples > 1:
        delta = sampler(n_residues=calpha_apo.shape[0], n_samples=n_samples, std=std).to(calpha_apo.device)
        delta -= delta.mean(dim=1, keepdim=True)
        candidates = calpha_apo.unsqueeze(0) + delta
        rmsds = ((candidates - calpha_holo.unsqueeze(0)) ** 2).sum(dim=-1).mean(dim=-1)
        return candidates[rmsds.argmin()]
    delta = sampler(n_residues=calpha_apo.shape[0], n_samples=1, std=std).to(calpha_apo.device)
    delta -= delta.mean(dim=0, keepdim=True)
    return calpha_apo + delta


def sample_prior_for_inference(complex_graph, n_samples: int, std: float, sampler):
    n_residues = complex_graph["receptor"].pos.shape[0]
    if n_samples > 1:
        calpha_holo = complex_graph["atom"].orig_holo_pos[complex_graph["atom"].ca_mask]
        delta = sampler(n_residues=n_residues, n_samples=n_samples, std=std).to(complex_graph["receptor"].pos.device)
        delta -= delta.mean(dim=1, keepdim=True)
        candidates = complex_graph["receptor"].pos.unsqueeze(0) + delta
        rmsds = ((candidates - calpha_holo.unsqueeze(0)) ** 2).sum(dim=-1).mean(dim=-1)
        return delta[rmsds.argmin()]
    delta = sampler(n_residues=n_residues, n_samples=1, std=std).to(complex_graph["receptor"].pos.device)
    delta -= delta.mean(dim=0, keepdim=True)
    return delta
