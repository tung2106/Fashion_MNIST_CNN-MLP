"""Đóng gói mã nguồn hiện tại để upload Colab; không kèm dữ liệu/trọng số."""

from zipfile import ZIP_DEFLATED, ZipFile

from utils.paths import ROOT


def main():
    archive = ROOT / "experiments" / "source.zip"
    archive.parent.mkdir(exist_ok=True)
    files = [ROOT / "requirements.txt", ROOT / "pyproject.toml"]
    for folder in [
        "data",
        "models",
        "training",
        "evaluation",
        "utils",
        "tests",
        "demo",
    ]:
        files.extend((ROOT / folder).glob("*.py"))
    files.extend((ROOT / "training/configs").glob("*.json"))
    # Kiểm tra hồi quy trên Colab đối chiếu kiến trúc với notebook nền tảng.
    files.append(ROOT / "notebooks/02_thuc_hanh_mlp_cnn.ipynb")
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        for path in sorted(files):
            output.write(path, path.relative_to(ROOT).as_posix())
    print(archive)


if __name__ == "__main__":
    main()
