from pathlib import Path


def run_workspace_dir(workspace_path: Path, run_number: int) -> Path:
    """Scratch folder that holds one .npz per processed event for this run."""
    d = workspace_path / f"run_{run_number}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def create_event_file(workspace_path: Path, run_number: int, idx: int)-> Path:
    file_name = run_workspace_dir(workspace_path, run_number) / f"event_{idx:06d}.npz"
    return file_name
