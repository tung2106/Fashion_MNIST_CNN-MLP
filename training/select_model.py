"""So sánh trên validation, chốt cấu hình rồi mới test. Chỉ chạy trên Colab GPU."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score

from data.dataset import load_test, make_loader, prepare_data
from evaluation.evaluate import evaluate_models, predict
from evaluation.plots import plot_history
from models.networks import MODEL_CLASSES
from training.engine import fit
from utils.io import file_hash, save_json
from utils.paths import DATA_DIR, ROOT, RUNS_DIR
from utils.runtime import require_colab_gpu

TRAINING_DIR = Path(__file__).resolve().parent


def trial(label, model_name, seed, augment, base, suite):
    run = suite / label
    for folder in ["checkpoints", "metrics", "figures"]:
        (run / folder).mkdir(parents=True)
    config = {**base, "models": [model_name], "seed": seed, "augment": augment}
    save_json(run / "config.json", config)
    data = prepare_data(DATA_DIR, config, run)
    info = fit(model_name, data, config, torch.device("cuda"), run)
    plot_history(model_name, run)
    payload = torch.load(
        run / "checkpoints" / f"{model_name}_best.pt", weights_only=True
    )
    model = MODEL_CLASSES[model_name]().cuda()
    model.load_state_dict(payload["state_dict"])
    truth, probs = predict(model, make_loader(data["validation"], config), "cuda")
    result = dict(
        trial=label,
        model=model_name,
        seed=seed,
        augment=augment,
        val_accuracy=accuracy_score(truth, probs.argmax(1)),
        val_macro_f1=f1_score(truth, probs.argmax(1), average="macro"),
        **info,
    )
    save_json(run / "validation.json", result)
    save_json(run / "STATUS.json", dict(stage="completed", test_evaluated=False))
    print("VALIDATION", json.dumps(result), flush=True)
    return result


def main():
    require_colab_gpu()
    torch.set_num_threads(2)
    base = json.loads((TRAINING_DIR / "configs/selection.json").read_text())
    suite = RUNS_DIR / (
        "selection_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    )
    suite.mkdir(parents=True)
    save_json(
        suite / "environment.json",
        dict(
            torch=str(torch.__version__),
            torchvision=str(torchvision.__version__),
            sklearn=sklearn.__version__,
            gpu=torch.cuda.get_device_name(0),
        ),
    )
    candidates = [
        ("mlp", "MLP", False),
        ("cnn", "CNN", False),
        ("cnn_dropout", "CNNDropout", False),
        ("cnn_batchnorm", "CNNBatchNorm", False),
        ("cnn_aug", "CNN", True),
        ("cnn_plus", "CNNPlus", False),
        ("cnn_plus_aug", "CNNPlus", True),
    ]
    save_json(
        suite / "protocol.json",
        dict(
            config=base,
            candidates=candidates,
            checkpoint_selection="minimum validation loss",
            config_selection="maximum validation macro F1; tie lower validation loss",
            confirm_seeds=[42, 123, 2026],
            demo_seed=42,
            test_note="Test đã được xem ở giai đoạn 2; không phải holdout chưa từng biết.",
        ),
    )
    # Lưu chính xác source đã chạy, không phụ thuộc các thay đổi GitHub sau đó.
    source = suite / "source"
    source.mkdir()
    for folder in ["data", "models", "training", "evaluation", "utils"]:
        shutil.copytree(
            ROOT / folder,
            source / folder,
            ignore=shutil.ignore_patterns("__pycache__", "raw", "checkpoints"),
        )
    (RUNS_DIR / "latest_selection.txt").write_text(suite.name)
    results = []
    try:
        for label, name, aug in candidates:
            results.append(trial(label, name, 42, aug, base, suite))
            pd.DataFrame(results).to_csv(
                suite / "validation_comparison.csv", index=False
            )
        winner = sorted(
            results, key=lambda r: (-r["val_macro_f1"], r["best_val_loss"])
        )[0]
        save_json(suite / "selected_config.json", winner)
        for seed in [123, 2026]:
            results.append(
                trial(
                    f"{winner['trial']}_seed{seed}",
                    winner["model"],
                    seed,
                    winner["augment"],
                    base,
                    suite,
                )
            )
            pd.DataFrame(results).to_csv(
                suite / "validation_comparison.csv", index=False
            )
        frozen = [
            r
            for r in results
            if r["trial"] in ["mlp", "cnn", winner["trial"]] or r["seed"] != 42
        ]
        hashes = {
            r["trial"]: file_hash(
                suite / r["trial"] / "checkpoints" / f"{r['model']}_best.pt"
            )
            for r in frozen
        }
        save_json(suite / "frozen_selection.json", dict(winner=winner, hashes=hashes))
        # Từ đây chỉ đánh giá, không thay đổi cấu hình theo test.
        test_rows = []
        for result in frozen:
            run, name = suite / result["trial"], result["model"]
            config = json.loads((run / "config.json").read_text())
            payload = torch.load(
                run / "checkpoints" / f"{name}_best.pt", weights_only=True
            )
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        (payload["mean"],), (payload["std"],)
                    ),
                ]
            )
            test = load_test(DATA_DIR, {"transform": transform})
            info = json.loads((run / "metrics" / f"{name}_training.json").read_text())
            evaluate_models(
                make_loader(test, config), config, "cuda", run, {name: info}
            )
            row = pd.read_csv(run / "metrics/summary.csv").iloc[0].to_dict()
            test_rows.append(dict(trial=result["trial"], seed=result["seed"], **row))
            save_json(run / "STATUS.json", dict(stage="completed", test_evaluated=True))
        pd.DataFrame(test_rows).to_csv(suite / "test_comparison.csv", index=False)
        selected_rows = [
            r for r in test_rows if r["trial"] == winner["trial"] or r["seed"] != 42
        ]
        save_json(
            suite / "stability.json",
            {
                metric: {
                    "mean": float(np.mean([r[metric] for r in selected_rows])),
                    "std": float(np.std([r[metric] for r in selected_rows], ddof=1)),
                }
                for metric in ["accuracy", "macro_f1"]
            },
        )
        demo = suite / "demo_bundle"
        demo.mkdir()
        shutil.copy2(
            suite / winner["trial"] / "checkpoints" / f"{winner['model']}_best.pt",
            demo / "model.pt",
        )
        shutil.copy2(ROOT / "models/networks.py", demo / "models.py")
        shutil.copy2(
            suite / winner["trial"] / "preprocessing.json", demo / "preprocessing.json"
        )
        save_json(
            demo / "metadata.json",
            dict(
                selection=winner,
                source_run=str(suite),
                checkpoint_sha256=file_hash(demo / "model.pt"),
                intended_input="Fashion-MNIST grayscale 28x28; ảnh thực tế chưa được kiểm chứng",
            ),
        )
        save_json(suite / "STATUS.json", dict(stage="completed", test_evaluated=True))
        print(
            "FINAL TEST\n", pd.DataFrame(test_rows).to_string(index=False), flush=True
        )
    except Exception as error:
        save_json(suite / "STATUS.json", dict(stage="failed", error=str(error)))
        raise
    finally:
        archive = shutil.make_archive(str(suite), "zip", suite)
        print("RESULT_ZIP:", archive, flush=True)


if __name__ == "__main__":
    main()
