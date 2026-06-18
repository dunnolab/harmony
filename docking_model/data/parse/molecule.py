import os
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith(".mol2"):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".sdf"):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith(".pdbqt"):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ""
        for line in pdbqt_data:
            pdb_block += "{}\n".format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError(
            "Expect the format of the molecule_file to be "
            "one of .mol2, .sdf, .pdbqt and .pdb, got {}".format(molecule_file)
        )

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn("Unable to compute charges for the molecule.")

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol


def read_mol(pdbbind_dir, name, remove_hs=False):
    lig = read_molecule(
        os.path.join(pdbbind_dir, name, f"{name}_ligand.sdf"),
        remove_hs=remove_hs,
        sanitize=True,
    )
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        lig = read_molecule(
            os.path.join(pdbbind_dir, name, f"{name}_ligand.mol2"),
            remove_hs=remove_hs,
            sanitize=True,
        )
    return lig


def read_single_mol(molecule_file, remove_hs=False):
    lig = read_molecule(
        molecule_file,
        remove_hs=remove_hs,
        sanitize=True,
    )
    if lig is None and molecule_file.endswith(".sdf"):
        mol2_file = molecule_file[:-4] + ".mol2"
        if os.path.exists(mol2_file):
            print(
                "Using the .sdf file failed. We found a .mol2 file instead and are trying to use that."
            )
            lig = read_molecule(
                mol2_file,
                remove_hs=remove_hs,
                sanitize=True,
            )
            if lig is None:
                print("Usage of .mol2 also failed.")
    return lig


def resolve_ligand_path(base_dir, name, ligand_input):
    candidates = []

    if ligand_input is None:
        return None

    ligand_input = str(ligand_input)
    if os.path.exists(ligand_input):
        return ligand_input

    if base_dir is not None and name is not None:
        candidates.append(os.path.join(base_dir, name, ligand_input))
    if base_dir is not None:
        candidates.append(os.path.join(base_dir, ligand_input))
    if name is not None:
        candidates.append(os.path.join(name, ligand_input))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve ligand input '{ligand_input}' relative to "
        f"base_dir='{base_dir}' and name='{name}'"
    )


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".sdf") and "rdkit" not in file:
            lig = read_single_mol(
                os.path.join(pdbbind_dir, name, file),
                remove_hs=remove_hs,
            )
            if lig is not None:
                ligs.append(lig)
    return ligs


def read_mols_v2(base_dir, remove_hs=False):
    ligs = []
    for file in os.listdir(base_dir):
        if file.endswith(".sdf") and "rdkit" not in file:
            lig = read_single_mol(
                os.path.join(base_dir, file),
                remove_hs=remove_hs,
            )
            if lig is not None:
                ligs.append(lig)
    return ligs


def read_sdf_or_mol2(sdf_fileName, mol2_fileName):
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception:
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            problem = False
        except Exception:
            problem = True

    return mol, problem


def write_mol_with_coords(mol, new_coords, path):
    w = Chem.SDWriter(path)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = new_coords.astype(np.double)[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))
    w.write(mol)
    w.close()
