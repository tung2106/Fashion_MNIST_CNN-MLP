"""Phân tích checkpoint đã chốt; không train hoặc chọn lại cấu hình."""

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Subset
from torchvision.datasets import FashionMNIST

from src.data import make_loader
from src.evaluate import predict
from src.inference import load_model, predict_image
from src.utils import file_hash, save_json

STAGE = Path(__file__).resolve().parent


def main(suite):
    selected = json.loads((suite / "selected_config.json").read_text())
    run = suite / selected["trial"]
    name = selected["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload, transform = load_model(
        run / "checkpoints" / f"{name}_best.pt", device
    )
    config = json.loads((run / "config.json").read_text())
    report = suite / "analysis"
    report.mkdir(exist_ok=True)
    raw = STAGE.parent / "du_lieu_chung/FashionMNIST/raw"
    save_json(
        report / "data_manifest.json",
        {p.name: file_hash(p) for p in sorted(raw.glob("*")) if p.is_file()},
    )
    log = STAGE.parent / "training.log"
    if log.exists():
        shutil.copy2(log, report / "training.log")
    train = FashionMNIST(
        STAGE.parent / "du_lieu_chung", train=True, transform=transform
    )
    test = FashionMNIST(
        STAGE.parent / "du_lieu_chung", train=False, transform=transform
    )
    ids = np.load(run / "splits.npz")
    clean = []
    for split, indexes in [
        ("train_clean", ids["train"]),
        ("validation", ids["validation"]),
    ]:
        truth, probs = predict(
            model, make_loader(Subset(train, indexes.tolist()), config), device
        )
        clean.append(
            dict(
                split=split,
                accuracy=accuracy_score(truth, probs.argmax(1)),
                macro_f1=f1_score(truth, probs.argmax(1), average="macro"),
            )
        )
    pd.DataFrame(clean).to_csv(report / "clean_scores.csv", index=False)
    table = pd.read_csv(run / "metrics" / f"{name}_predictions.csv")
    matrix = confusion_matrix(table.true_label, table.predicted_label, labels=range(10))
    saved = np.load(run / "metrics" / f"{name}_confusion.npy")
    assert np.array_equal(matrix, saved) and matrix.sum() == 10000
    scores = json.loads(
        (run / "metrics" / f"{name}_classification_report.json").read_text()
    )
    assert np.isclose(matrix.trace() / matrix.sum(), scores["accuracy"])
    assert all(
        matrix[k].sum() == scores[c]["support"]
        for k, c in enumerate(payload["classes"])
    )
    assert table.checkpoint_hash.nunique() == 1
    assert table.checkpoint_hash.iloc[0] == file_hash(
        run / "checkpoints" / f"{name}_best.pt"
    )
    pairs = sorted(
        [(int(matrix[a, b]), a, b) for a in range(10) for b in range(10) if a != b],
        reverse=True,
    )
    pd.DataFrame(
        [
            dict(
                true_label=payload["classes"][a],
                predicted_label=payload["classes"][b],
                count=n,
            )
            for n, a, b in pairs[:10]
        ]
    ).to_csv(report / "top_confusions.csv", index=False)
    table["confidence"] = table[[f"prob_{i}" for i in range(10)]].max(axis=1)
    wrong = table[table.true_label != table.predicted_label].sort_values(
        "confidence", ascending=False
    )
    wrong.head(12).to_csv(report / "error_samples.csv", index=False)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for ax, row in zip(axes.flat, wrong.head(12).itertuples()):
        ax.imshow(test.data[row.sample_id], cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"ID {row.sample_id}: {payload['classes'][row.true_label]}\n→ {payload['classes'][row.predicted_label]} ({row.confidence:.1%})",
            fontsize=9,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(report / "confident_errors.png")
    plt.close(fig)
    # Ảnh trung bình mô tả hình dáng lớp, không phải feature engineering mới.
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for k, ax in enumerate(axes.flat):
        chosen = ids["train"][train.targets[ids["train"]].numpy() == k]
        ax.imshow(train.data[chosen].float().mean(0), cmap="gray", vmin=0, vmax=255)
        ax.set_title(payload["classes"][k])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(report / "train_class_means.png")
    plt.close(fig)
    sample_ids = [
        int(table[table.true_label == k].iloc[0].sample_id) for k in [0, 2, 4, 6]
    ]
    sample_ids += wrong.head(2).sample_id.astype(int).tolist()
    feature_rows = []
    for sample_id in dict.fromkeys(sample_ids):
        image = Image.fromarray(test.data[sample_id].numpy())
        probs = predict_image(model, image, transform, device).numpy()
        expected = table.loc[
            table.sample_id == sample_id, [f"prob_{i}" for i in range(10)]
        ].values[0]
        np.testing.assert_allclose(probs, expected, atol=2e-5, rtol=2e-4)
        image.save(report / f"sample_{sample_id}.png")
        if not hasattr(model, "features"):
            continue
        maps, x = [], transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            for layer_id, layer in enumerate(model.features):
                x = layer(x)
                if isinstance(layer, torch.nn.ReLU):
                    maps.append((layer_id, x[0, :4].cpu().numpy()))
        fig, axes = plt.subplots(2, 5, figsize=(11, 5))
        for row, (layer_id, activation) in enumerate([maps[0], maps[-1]]):
            axes[row, 0].imshow(image, cmap="gray")
            axes[row, 0].set_title(f"Test ID {sample_id}")
            for channel in range(4):
                axes[row, channel + 1].imshow(activation[channel], cmap="viridis")
                axes[row, channel + 1].set_title(f"Layer {layer_id}, ch {channel}")
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(report / f"features_{sample_id}.png")
        plt.close(fig)
        feature_rows.append(
            dict(
                sample_id=sample_id,
                layers=[maps[0][0], maps[-1][0]],
                channels=[0, 1, 2, 3],
            )
        )
    save_json(
        report / "feature_metadata.json",
        dict(
            checkpoint_sha256=file_hash(run / "checkpoints" / f"{name}_best.pt"),
            split="test",
            samples=feature_rows,
            color_scale="independent per channel",
            causal_interpretation=False,
        ),
    )
    architecture, handles = [], []

    def record(layer_name):
        def hook(layer, inputs, output):
            architecture.append(
                dict(
                    layer=layer_name,
                    type=type(layer).__name__,
                    shape=list(output.shape),
                    parameters=sum(p.numel() for p in layer.parameters(recurse=False)),
                )
            )

        return hook

    for layer_name, layer in model.named_modules():
        if not list(layer.children()):
            handles.append(layer.register_forward_hook(record(layer_name)))
    sample = transform(Image.fromarray(test.data[0].numpy())).unsqueeze(0).to(device)
    with torch.inference_mode():
        model(sample)
    for handle in handles:
        handle.remove()
    save_json(report / "architecture.json", architecture)
    timings = []
    with torch.inference_mode():
        for i in range(110):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(sample)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= 10:
                timings.append((time.perf_counter() - start) * 1000)
    save_json(
        report / "latency.json",
        dict(
            device=device,
            batch_size=1,
            warmup=10,
            measured=100,
            median_ms=float(np.median(timings)),
            p95_ms=float(np.percentile(timings, 95)),
            includes_preprocessing=False,
        ),
    )
    save_json(
        report / "checks.json",
        dict(
            predictions_matrix_accuracy_support="PASS",
            checkpoint_hash="PASS",
            inference_matches_evaluation="PASS",
            train_validation_overlap=int(
                np.intersect1d(ids["train"], ids["validation"]).size
            ),
        ),
    )
    shutil.copy2(__file__, report / "analyze_selection.py")
    shutil.copy2(STAGE / "src/inference.py", report / "inference.py")
    print("ANALYSIS_COMPLETE", report)
    archive = shutil.make_archive(str(suite), "zip", suite)
    print("RESULT_ZIP:", archive)
    print("SHA256:", file_hash(Path(archive)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None)
    args = parser.parse_args()
    suite = args.run or Path((STAGE / "artifacts/latest_selection.txt").read_text())
    main(suite)
