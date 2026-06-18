import numpy as np
import torch
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch

from docking_model.geometry.ops import (
    rigid_transform_kabsch,
    axis_angle_to_matrix,
    rigid_transform_kabsch_batch,
)


def modify_conformer_torsion_angles(
    pos,
    edge_index,
    mask_rotate,
    torsion_updates,
    fragment_index=None,
    as_numpy=False,
    sidechains: bool = False,
):
    if fragment_index is None:
        raise ValueError("fragment_index is required for graph-native torsion updates.")
    if as_numpy:
        raise ValueError("as_numpy=True is not supported by graph-native torsion updates.")
    return modify_conformer_torsion_angles_new(
        pos=pos,
        edge_index=edge_index,
        mask_rotate=mask_rotate,
        fragment_index=fragment_index,
        torsion_angle_updates=torsion_updates,
        sidechains=sidechains,
    )


def modify_conformer_torsion_angles_new(
    pos,
    edge_index, 
    mask_rotate,
    fragment_index,
    torsion_angle_updates,
    sidechains: bool = False,
):
    assert edge_index.shape[0] == 2, "modify_conformer_torsion_angles_new expects edge indexes as [2, E]"

    pos = pos.clone()
    affine_mats = torch.eye(4, device=pos.device).repeat(
        *pos.shape[:-2], edge_index.shape[1] + 1, 1, 1
    )

    inv_tr_affine_mats = torch.eye(4, device=pos.device).repeat(
        *pos.shape[:-2], edge_index.shape[1] + 1, 1, 1
    )
    inv_tr_affine_mats[..., :-1, :-1, -1] = -1 * pos.index_select(-2, edge_index[0])

    tr_affine_mats = torch.eye(4, device=pos.device).repeat(
        *pos.shape[:-2], edge_index.shape[1] + 1, 1, 1
    )
    tr_affine_mats[..., :-1, :-1, -1] = pos.index_select(-2, edge_index[0])

    torsion_axis_angle = pos.index_select(
        -2, edge_index[:, mask_rotate][1]
    ) - pos.index_select(-2, edge_index[:, mask_rotate][0])
    torsion_axis_angle = (
        torsion_axis_angle
        / torch.linalg.norm(torsion_axis_angle, dim=-1, keepdims=True)
        * torsion_angle_updates.unsqueeze(-1)
    )
    torsion_affine_mats = torch.eye(4, device=pos.device).repeat(
        *pos.shape[:-2], edge_index.shape[1] + 1, 1, 1
    )

    torsion_affine_mats[
        ...,
        torch.cat((mask_rotate, torch.tensor([False], device=mask_rotate.device))),
        :-1,
        :-1,
    ] = axis_angle_to_matrix(torsion_axis_angle)

    # In the old version, sidechain modifications used a rotmat.T when applying the updates
    if sidechains:
        affine_mats = (
            tr_affine_mats @ torsion_affine_mats.transpose(-1, -2) @ inv_tr_affine_mats
        )
    else:
        affine_mats = tr_affine_mats @ torsion_affine_mats @ inv_tr_affine_mats
    affine_mats = affine_mats.swapaxes(-1, -2)

    atom_transforms = to_dense_batch(
        fragment_index[0],
        fragment_index[1],
        fill_value=edge_index.shape[1],
        batch_size=pos.shape[-2],
    )[0].T

    pos = torch.cat((pos, pos.new_ones((*pos.shape[:-1], 1))), dim=-1).unsqueeze(-2)
    for atom_transform_slice in atom_transforms:
        pos = pos @ affine_mats.index_select(-3, atom_transform_slice)
    pos = pos[..., :-1].squeeze(-2)

    return pos


def modify_conformer_fast(data, tr_update, rot_update, torsion_updates, pivot=None):
    lig_center = torch.mean(data["ligand"].pos, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = (data["ligand"].pos - lig_center) @ rot_mat + tr_update + lig_center

    if torsion_updates is not None:
        flexible_new_pos = modify_conformer_torsion_angles(
            pos=rigid_new_pos,
            edge_index=data["ligand", "lig_bond", "ligand"].edge_index,
            mask_rotate=data["ligand"].edge_mask,
            fragment_index=data["ligand"].lig_fragment_index,
            torsion_updates=torch.from_numpy(torsion_updates).float()
            if isinstance(torsion_updates, np.ndarray)
            else torsion_updates,
        )

        if pivot is None:
            R, t = rigid_transform_kabsch(flexible_new_pos, rigid_new_pos)
            aligned_flexible_pos = (flexible_new_pos @ R.T) + t
        else:
            R1, t1 = rigid_transform_kabsch(pivot, rigid_new_pos)
            R2, t2 = rigid_transform_kabsch(flexible_new_pos, pivot)

            aligned_flexible_pos = (flexible_new_pos @ R2.T + t2) @ R1.T + t1

        data["ligand"].pos = aligned_flexible_pos
    else:
        data["ligand"].pos = rigid_new_pos
    return data


def modify_conformer_fast_batch(
    data, tr_update, rot_update, torsion_updates, pivot=None
):
    batch = data["ligand"].batch

    lig_center = scatter_mean(data["ligand"].pos, index=batch, dim=0)  # (B x 3)
    rot_mat = axis_angle_to_matrix(rot_update)  # (B x 3 x 3)

    rigid_new_pos = (
        torch.einsum(
            "ij,ijk->ik", (data["ligand"].pos - lig_center[batch]), rot_mat[batch]
        )
        + lig_center[batch]
        + tr_update[batch]
    )

    if torsion_updates is not None:
        flexible_new_pos = modify_conformer_torsion_angles(
            pos=rigid_new_pos,
            edge_index=data["ligand", "lig_bond", "ligand"].edge_index,
            mask_rotate=data["ligand"].edge_mask,
            fragment_index=data["ligand"].lig_fragment_index,
            torsion_updates=torch.from_numpy(torsion_updates).float()
            if isinstance(torsion_updates, np.ndarray)
            else torsion_updates,
        )

        if pivot is None:
            R, t = rigid_transform_kabsch_batch(
                flexible_new_pos, rigid_new_pos, batch=batch
            )
            aligned_flexible_pos = (
                torch.einsum("ij,ijk->ik", flexible_new_pos, R[batch].transpose(-1, -2))
                + t[batch]
            )
        else:
            raise ValueError()
        data["ligand"].pos = aligned_flexible_pos
    else:
        data["ligand"].pos = rigid_new_pos
    return data


def modify_conformer_coordinates_new(
    pos,
    edge_index,
    rotate_mask,
    fragment_index,
    batch,  # For ligand this should be the PyG batch idx and for protein this should be residue idx
    pivot_mask=None,  # For ligand this should be None and for protein this should be ca_mask
    align=False,  # For ligand this should be True and for protein this should be False
    tr_updates=None,
    rot_updates=None,
    torsion_updates=None,
):
    pos = pos.clone()
    if tr_updates is not None:
        pos += tr_updates[batch]

    if rot_updates is not None:
        rot_mats = axis_angle_to_matrix(rot_updates)
        pivots = (
            pos[pivot_mask]
            if pivot_mask is not None
            else scatter_mean(pos, batch, dim=0)
        )
        pos = (
            (pos - pivots[batch]).unsqueeze(-2) @ rot_mats[batch].swapaxes(-1, -2)
            + pivots[batch].unsqueeze(-2)
        ).squeeze()

    if torsion_updates is not None:
        flexible_pos = modify_conformer_torsion_angles_new(
            pos, edge_index, rotate_mask, fragment_index, torsion_updates
        )
        if align:
            R, tr = rigid_transform_kabsch_batch(flexible_pos, pos, batch)
            pos = (
                flexible_pos.unsqueeze(-2) @ R[batch].swapaxes(-1, -2)
                + tr[batch].unsqueeze(-2)
            ).squeeze()
        else:
            pos = flexible_pos

    return pos
