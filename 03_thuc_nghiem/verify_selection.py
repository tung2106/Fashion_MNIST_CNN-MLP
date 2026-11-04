"""Kiểm tra tính đúng trên GPU Colab; không phải đo chất lượng mô hình."""

import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.data import make_loader, split_indices
from src.engine import fit, run_epoch
from src.inference import load_model
from src.models import MODEL_CLASSES


def main():
    assert importlib.util.find_spec("google.colab") and torch.cuda.is_available()
    train, val = split_indices(np.repeat(np.arange(10), 6000), 42)
    assert len(train) == 54000 and len(val) == 6000
    assert not np.intersect1d(train, val).size
    assert np.array_equal(train, split_indices(np.repeat(np.arange(10), 6000), 42)[0])
    data = TensorDataset(torch.randn(8, 1, 28, 28), torch.arange(8))
    config = dict(seed=42, batch_size=4, lr=0.001, epochs=1)
    loader = make_loader(data, config)
    for name, factory in MODEL_CLASSES.items():
        model = factory().cuda()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        measured = run_epoch(model, loader, "cuda")
        assert np.isfinite(measured["loss"])
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        run_epoch(model, loader, "cuda", optimizer)
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
        assert model(torch.randn(2, 1, 28, 28, device="cuda")).shape == (2, 10)
    with tempfile.TemporaryDirectory() as folder:
        run = Path(folder)
        (run / "checkpoints").mkdir()
        (run / "metrics").mkdir()
        bundle = dict(
            train=data,
            validation=data,
            mean=0.3,
            std=0.3,
            classes=list(map(str, range(10))),
        )
        config.update(epochs=10, patience=3, min_delta=0.0001, scheduler=True)
        values = []
        for val_loss in [0.4, 0.3, 0.31, 0.32, 0.33]:
            values.extend(
                [dict(loss=0.4, accuracy=0.5), dict(loss=val_loss, accuracy=0.5)]
            )
        with patch("src.engine.run_epoch", side_effect=values):
            info = fit("CNNPlus", bundle, config, "cuda", run)
        assert info["epochs_run"] == 5 and info["best_epoch"] == 2
        model, payload, _ = load_model(run / "checkpoints/CNNPlus_best.pt", "cuda")
        assert payload["epoch"] == 2
        last = torch.load(run / "checkpoints/CNNPlus_last.pt", weights_only=True)
        assert last["optimizer"]["param_groups"][0]["lr"] == 0.0005
        expected = MODEL_CLASSES["CNNPlus"]().cuda().eval()
        expected.load_state_dict(payload["state_dict"])
        sample = torch.randn(2, 1, 28, 28, device="cuda")
        with torch.no_grad():
            torch.testing.assert_close(model(sample), expected(sample))
    print(
        "PASS: split, shapes, train/eval, early stopping, scheduler, checkpoint reload."
    )


if __name__ == "__main__":
    main()
