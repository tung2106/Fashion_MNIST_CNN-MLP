"""Đối chiếu kết quả và chạy lại suy luận test trên CPU, không huấn luyện."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from data.chuan_bi_du_lieu import split_indices
from evaluation.danh_gia_mo_hinh import predict
from models.du_doan import load_model, predict_image
from models.kien_truc_mo_hinh import MODEL_CLASSES
from utils.duong_dan import DATA_DIR, ROOT, latest_run
from utils.ho_tro_thuc_nghiem import file_hash


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_report(actual, expected):
    for key, value in expected.items():
        if isinstance(value, dict):
            compare_report(actual[key], value)
        else:
            assert np.isclose(actual[key], value, atol=1e-12), (key, actual[key], value)


def verify_saved_metrics(run, raw_test, classes):
    rows = []
    table = pd.read_csv(run / "test_comparison.csv")
    for result in table.to_dict("records"):
        trial = run / result["trial"]
        name = result["model"]
        prediction = pd.read_csv(trial / "metrics" / f"{name}_predictions.csv")
        expected_hash = file_hash(trial / "checkpoints" / f"{name}_best.pt")

        np.testing.assert_array_equal(prediction["sample_id"], np.arange(10000))
        np.testing.assert_array_equal(
            prediction["true_label"], raw_test.targets.numpy()
        )
        assert prediction["checkpoint_hash"].eq(expected_hash).all()
        probabilities = prediction[[f"prob_{k}" for k in range(10)]].to_numpy()
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=2e-6)
        predicted = probabilities.argmax(1)
        np.testing.assert_array_equal(predicted, prediction["predicted_label"])

        truth = prediction["true_label"].to_numpy()
        matrix = confusion_matrix(truth, predicted, labels=range(10))
        np.testing.assert_array_equal(
            matrix, np.load(trial / "metrics" / f"{name}_confusion.npy")
        )
        np.testing.assert_array_equal(matrix.sum(1), [1000] * 10)
        accuracy = accuracy_score(truth, predicted)
        macro_f1 = f1_score(truth, predicted, average="macro")
        assert np.isclose(accuracy, result["accuracy"], atol=1e-12)
        assert np.isclose(macro_f1, result["macro_f1"], atol=1e-12)
        assert matrix.trace() / 10000 == accuracy
        compare_report(
            classification_report(
                truth,
                predicted,
                target_names=classes,
                output_dict=True,
                zero_division=0,
            ),
            read_json(trial / "metrics" / f"{name}_classification_report.json"),
        )
        rows.append(
            dict(
                trial=result["trial"],
                accuracy=accuracy,
                macro_f1=macro_f1,
                status="PASS",
            )
        )
    return rows


def verify_test_inference(run, registry):
    rows = []
    for key, entry in registry.items():
        checkpoint = ROOT / "models" / entry["checkpoint"]
        assert file_hash(checkpoint) == entry["sha256"]
        model, payload, transform = load_model(checkpoint)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        dataset = FashionMNIST(
            DATA_DIR, train=False, download=False, transform=transform
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        started = time.perf_counter()
        truth, probabilities = predict(model, loader, "cpu")
        seconds = time.perf_counter() - started

        stored = pd.read_csv(
            run / entry["trial"] / "metrics" / f"{entry['model']}_predictions.csv"
        )
        expected = stored[[f"prob_{k}" for k in range(10)]].to_numpy()
        np.testing.assert_allclose(probabilities, expected, atol=2e-5, rtol=2e-4)
        np.testing.assert_array_equal(truth, stored["true_label"])
        np.testing.assert_array_equal(
            probabilities.argmax(1), stored["predicted_label"]
        )
        assert np.isfinite(probabilities).all()
        assert not model.training
        for name, value in model.state_dict().items():
            assert torch.equal(value, before[name]), name

        for size, mode in [((28, 28), "RGB"), ((27, 28), "L")]:
            try:
                predict_image(model, Image.new(mode, size), transform)
            except ValueError:
                pass
            else:
                raise AssertionError("Ảnh không đúng định dạng phải bị từ chối")
        one = predict_image(
            model, Image.fromarray(dataset.data[183].numpy()), transform
        )
        two = predict_image(
            model, Image.fromarray(dataset.data[183].numpy()), transform
        )
        assert torch.equal(one, two)
        np.testing.assert_allclose(
            one.numpy(), probabilities[183], atol=2e-5, rtol=2e-4
        )

        row = dict(
            model=entry["model"],
            checkpoint=entry["checkpoint"],
            samples=len(truth),
            correct=int((truth == probabilities.argmax(1)).sum()),
            accuracy=float(accuracy_score(truth, probabilities.argmax(1))),
            macro_f1=float(f1_score(truth, probabilities.argmax(1), average="macro")),
            max_probability_difference=float(np.max(np.abs(probabilities - expected))),
            inference_seconds=seconds,
            parameters=sum(p.numel() for p in model.parameters()),
            state_unchanged=True,
            invalid_images_rejected=True,
            repeatable_prediction=True,
            sha256=entry["sha256"],
            status="PASS",
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = latest_run()
    assert read_json(run / "STATUS.json")["stage"] == "completed"
    raw_train = FashionMNIST(DATA_DIR, train=True, download=False)
    raw_test = FashionMNIST(DATA_DIR, train=False, download=False)
    assert len(raw_train) == 60000 and len(raw_test) == 10000
    train_ids, val_ids = split_indices(raw_train.targets, 42)
    np.testing.assert_array_equal(
        np.bincount(raw_train.targets[train_ids]), [5400] * 10
    )
    np.testing.assert_array_equal(np.bincount(raw_train.targets[val_ids]), [600] * 10)

    hashes = read_json(ROOT / "reports/results/20261104/data_manifest.json")
    for filename, expected in hashes.items():
        assert file_hash(DATA_DIR / "FashionMNIST/raw" / filename) == expected
    immutable = [p for p in run.rglob("*") if p.is_file()]
    immutable += list((ROOT / "models/checkpoints").glob("*.pt"))
    before_hashes = {path: file_hash(path) for path in immutable}

    pixels = raw_train.data[train_ids].float() / 255.0
    mean, std = pixels.mean().item(), pixels.std(unbiased=False).item()
    del pixels
    validation = pd.read_csv(run / "validation_comparison.csv")
    for row in validation.to_dict("records"):
        trial = run / row["trial"]
        split = np.load(trial / "splits.npz")
        np.testing.assert_array_equal(train_ids, split["train"])
        np.testing.assert_array_equal(val_ids, split["validation"])
        preprocessing = read_json(trial / "preprocessing.json")
        assert abs(mean - preprocessing["mean"]) < 1e-6
        assert abs(std - preprocessing["std"]) < 1e-6
        history = pd.read_csv(trial / "metrics" / f"{row['model']}_history.csv")
        best = history.loc[history["val_loss"].idxmin()]
        assert int(best["epoch"]) == int(row["best_epoch"])
        assert np.isclose(best["val_loss"], row["best_val_loss"], atol=1e-12)
    candidates = validation[validation["seed"] == 42]
    winner = candidates.sort_values(
        ["val_macro_f1", "best_val_loss"], ascending=[False, True]
    ).iloc[0]
    assert winner["trial"] == read_json(run / "selected_config.json")["trial"]

    architectures = {}
    for name, constructor in MODEL_CLASSES.items():
        model = constructor().eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 1, 28, 28))
        assert output.shape == (2, 10) and torch.isfinite(output).all()
        architectures[name] = list(output.shape)

    saved = verify_saved_metrics(run, raw_test, raw_train.classes)
    selected_trials = [
        row for row in saved if row["trial"].startswith(str(winner["trial"]))
    ]
    assert len(selected_trials) == 3
    stability = {
        metric: {
            "mean": float(np.mean([row[metric] for row in selected_trials])),
            "std": float(np.std([row[metric] for row in selected_trials], ddof=1)),
        }
        for metric in ["accuracy", "macro_f1"]
    }
    compare_report(stability, read_json(run / "stability.json"))
    report_dir = ROOT / "reports/results/20261104"
    for filename in [
        "test_comparison.csv",
        "validation_comparison.csv",
        "stability.json",
        "protocol.json",
        "environment.json",
    ]:
        assert file_hash(report_dir / filename) == file_hash(run / filename)
    assert file_hash(report_dir / "CNNPlus_classification_report.json") == file_hash(
        run / winner["trial"] / "metrics/CNNPlus_classification_report.json"
    )
    registry = read_json(ROOT / "models/danh_sach_mo_hinh.json")
    inference = verify_test_inference(run, registry)
    for path, expected in before_hashes.items():
        assert file_hash(path) == expected, path

    report = dict(
        checked_at=datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        source_run=run.name,
        training_executed=False,
        device="cpu",
        threads=2,
        data=dict(
            train=54000,
            validation=6000,
            test=10000,
            mean=mean,
            std=std,
            class_counts="5400/600/1000 per class",
            raw_hashes_verified=len(hashes),
            matching_trial_splits=len(validation),
        ),
        validation_selection=winner["trial"],
        architectures=architectures,
        saved_test_metrics=saved,
        stability=stability,
        curated_report_tables_match=True,
        full_test_inference=inference,
        immutable_files_verified=len(before_hashes),
        status="PASS",
        limits=[
            "No training/optimizer/GPU regression executed locally.",
            "Test set was already inspected earlier; not a fresh unseen holdout.",
            "Only Fashion-MNIST grayscale 28x28, not real camera photos.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"VERIFIED: {args.output}", flush=True)


if __name__ == "__main__":
    main()
