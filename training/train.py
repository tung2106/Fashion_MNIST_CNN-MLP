"""Điểm bắt đầu: đọc cấu hình -> dữ liệu -> train -> tùy chọn test."""

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
import torchvision

from data.dataset import load_test, make_loader, prepare_data
from evaluation.evaluate import evaluate_models
from evaluation.plots import plot_history, plot_samples
from models.networks import MODEL_CLASSES
from training.engine import fit
from utils.io import save_json
from utils.paths import DATA_DIR, RUNS_DIR
from utils.runtime import require_colab_gpu

TRAINING_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/baseline.json",
        help="Tương đối với thư mục training/.",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Chỉ xem dữ liệu, không train."
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Test sau khi train và chọn best bằng validation.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    config = json.loads((TRAINING_DIR / args.config).read_text(encoding="utf-8"))
    if config["profile"] not in ("baseline", "smoke"):
        parser.error("profile phải là baseline hoặc smoke")
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["lr"] <= 0:
        parser.error("epochs, batch_size, lr phải dương")
    if not config["models"] or any(
        name not in MODEL_CLASSES for name in config["models"]
    ):
        parser.error(f"models phải thuộc {list(MODEL_CLASSES)}")
    if args.evaluate_test and (config["profile"] == "smoke" or args.inspect):
        parser.error("Không test trong chế độ smoke/inspect.")
    # Giữ số thread nhất quán với cấu hình thực nghiệm.
    torch.set_num_threads(2)
    if not args.inspect:
        require_colab_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = RUNS_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    for name in ["checkpoints", "metrics", "figures"]:
        (run / name).mkdir()
    config.update(
        device=str(device),
        python=platform.python_version(),
        torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),
        evaluate_test=args.evaluate_test,
        selection="min_validation_loss",
        data_dir=str(args.data_dir.resolve()),
    )
    save_json(run / "config.json", config)
    save_json(run / "STATUS.json", {"stage": "preparing"})
    print("Run:", run, flush=True)
    try:
        data = prepare_data(args.data_dir, config, run)
        images, labels = next(iter(make_loader(data["train"], config)))
        print("Train/validation:", len(data["train"]), len(data["validation"]))
        print("Batch images/labels:", tuple(images.shape), tuple(labels.shape))
        plot_samples(data, run)
        if args.inspect:
            save_json(
                run / "STATUS.json",
                {"stage": "data_inspected", "trained": False, "test_evaluated": False},
            )
            return
        training_info = {}
        for name in config["models"]:
            training_info[name] = fit(name, data, config, device, run)
            plot_history(name, run)
        if args.evaluate_test:
            test = load_test(args.data_dir, data)
            evaluate_models(
                make_loader(test, config), config, device, run, training_info
            )
        save_json(
            run / "STATUS.json",
            {
                "stage": "completed",
                "profile": config["profile"],
                "test_evaluated": args.evaluate_test,
                "limits": ["one seed", "no ablation", "no exact resume"],
            },
        )
        print("COMPLETE:", run, flush=True)
    except Exception as error:
        save_json(run / "STATUS.json", {"stage": "failed", "error": str(error)})
        raise


if __name__ == "__main__":  # Chạy khi gọi file; không tự chạy khi import.
    main()
