"""Đường dẫn chung tính từ vị trí dự án, không phụ thuộc thư mục terminal."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RUNS_DIR = ROOT / "experiments" / "runs"
DEFAULT_CHECKPOINT = ROOT / "models" / "checkpoints" / "cnn_plus.pt"
SELECTED_RUN_ID = "selection_20261104_144952_e58647"


def latest_run():
    pointer = RUNS_DIR / "latest_selection.txt"
    name = (
        pointer.read_text(encoding="utf-8").strip()
        if pointer.exists()
        else SELECTED_RUN_ID
    )
    path = Path(name)
    return path if path.is_absolute() else RUNS_DIR / path
