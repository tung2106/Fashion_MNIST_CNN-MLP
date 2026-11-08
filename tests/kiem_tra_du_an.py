"""Kiểm tra hợp đồng dữ liệu/checkpoint sau đổi cấu trúc; không huấn luyện."""

import ast
import importlib
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import FashionMNIST

from models.du_doan import load_model, predict_image
from utils.duong_dan import DATA_DIR, ROOT, latest_run
from utils.ho_tro_thuc_nghiem import file_hash
from utils.kiem_tra_gpu import is_colab_gpu, require_colab_gpu


def main():
    torch.set_num_threads(2)
    for module in [
        "models.kien_truc_mo_hinh",
        "models.du_doan",
        "data.chuan_bi_du_lieu",
        "training.huan_luyen",
        "training.vong_lap_huan_luyen",
        "training.chon_mo_hinh",
        "evaluation.danh_gia_mo_hinh",
        "evaluation.ve_bieu_do",
        "evaluation.phan_tich_ket_qua",
        "demo.du_doan_anh",
        "scripts.dong_goi_colab",
        "utils.duong_dan",
    ]:
        importlib.import_module(module)
    for folder in [
        "models",
        "data",
        "training",
        "evaluation",
        "demo",
        "scripts",
        "utils",
        "tests",
    ]:
        for path in (ROOT / folder).glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=path.name)

    if not is_colab_gpu():
        try:
            require_colab_gpu()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Guard huấn luyện không hoạt động.")

    train = FashionMNIST(DATA_DIR, train=True, download=False)
    test = FashionMNIST(DATA_DIR, train=False, download=False)
    assert len(train) == 60000 and len(test) == 10000
    run = latest_run()
    split = np.load(run / "cnn_plus_aug/splits.npz")
    assert len(split["train"]) == 54000 and len(split["validation"]) == 6000
    assert not np.intersect1d(split["train"], split["validation"]).size

    registry = json.loads((ROOT / "models/danh_sach_mo_hinh.json").read_text())
    samples = json.loads((ROOT / "demo/examples/labels.json").read_text())
    checked = []
    for key, entry in registry.items():
        checkpoint = ROOT / "models" / entry["checkpoint"]
        original = run / entry["trial"] / "checkpoints" / f"{entry['model']}_best.pt"
        assert file_hash(checkpoint) == file_hash(original)
        model, payload, transform = load_model(checkpoint)
        assert payload["model_name"] == entry["model"]
        predictions = pd.read_csv(
            run / entry["trial"] / "metrics" / f"{entry['model']}_predictions.csv"
        ).set_index("sample_id")
        for sample in samples:
            sid = sample["sample_id"]
            with Image.open(ROOT / f"demo/examples/sample_{sid}.png") as image:
                probs = predict_image(model, image, transform).numpy()
            expected = predictions.loc[sid, [f"prob_{i}" for i in range(10)]].to_numpy(
                dtype=float
            )
            np.testing.assert_allclose(probs, expected, atol=2e-5, rtol=2e-4)
        checked.append(
            dict(model=key, samples=len(samples), sha256=file_hash(checkpoint))
        )

    receipt = dict(
        imports="PASS",
        notebook_syntax="PASS",
        raw_samples=[len(train), len(test)],
        split_overlap=0,
        training_guard="PASS",
        inference_vs_original_predictions="PASS",
        checkpoints=checked,
        training_executed=False,
    )
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    main()
