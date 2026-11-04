"""Đóng mã nhỏ gọn để upload Colab. Không chạy mô hình hoặc kèm dữ liệu."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

stage = Path(__file__).resolve().parents[1]
archive = stage / "artifacts/source.zip"
archive.parent.mkdir(exist_ok=True)
files = [
    stage / name
    for name in [
        "main.py",
        "select_model.py",
        "analyze_selection.py",
        "verify_selection.py",
    ]
]
files += list((stage / "src").glob("*.py"))
files += list((stage / "configs").glob("*.json"))
with ZipFile(archive, "w", ZIP_DEFLATED) as output:
    for source in files:
        output.write(source, source.relative_to(stage.parent).as_posix())
print(archive)
