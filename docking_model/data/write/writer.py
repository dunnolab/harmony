from __future__ import annotations

import pickle
from pathlib import Path

from docking_model.data.write.trajectory import write_docking_trajectory_frames


def write_docking_outputs(
    docking_outputs: dict,
    output_dir: str | Path,
    *,
    export_trajectory_files: bool = True,
    trajectory_max_ranks: int | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "docking_predictions.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(docking_outputs, handle)
    if export_trajectory_files and (
        docking_outputs.get("ligand_trajectory") is not None
        or docking_outputs.get("atom_trajectory") is not None
    ):
        write_docking_trajectory_frames(
            docking_outputs=docking_outputs,
            output_dir=output_dir,
            max_ranks=trajectory_max_ranks,
        )
    return output_path
