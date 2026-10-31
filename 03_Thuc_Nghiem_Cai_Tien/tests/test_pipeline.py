"""Kiểm tra tính đúng; tensor giả chỉ dùng test kỹ thuật, không báo accuracy đồ án."""
import ast
from pathlib import Path
import unittest
import tempfile
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.models import MLP, CNN
from src.data import split_indices
from src.engine import run_epoch
from src.evaluate import predict, evaluate_models

torch.set_num_threads(2)


class PipelineTests(unittest.TestCase):
    def test_split_reproducible_disjoint_balanced(self):
        labels = np.repeat(np.arange(10), 6000)
        train, val = split_indices(labels, 42)
        train2, val2 = split_indices(labels, 42)
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(val, val2)
        self.assertEqual(len(set(train) | set(val)), 60000)
        self.assertFalse(set(train) & set(val))
        np.testing.assert_array_equal(np.bincount(labels[train]), [5400]*10)
        np.testing.assert_array_equal(np.bincount(labels[val]), [600]*10)

    def test_models_match_notebook_with_same_weights(self):
        path = Path(__file__).resolve().parents[2] / '02_Colab_Co_Ban' / 'ma_nguon_cac_o.py'
        parsed = ast.parse(path.read_text(encoding='utf-8'))
        definitions = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.ClassDef)
                                      and n.name in ('MLP', 'CNN')], type_ignores=[])
        scope = {'nn': nn}
        exec(compile(definitions, str(path), 'exec'), scope)
        for cls in [MLP, CNN]:
            original, migrated = scope[cls.__name__](), cls()
            migrated.load_state_dict(original.state_dict())
            images = torch.randn(3, 1, 28, 28)
            torch.testing.assert_close(original(images), migrated(images), rtol=0, atol=0)

    def test_training_changes_weights_evaluation_does_not(self):
        loader = DataLoader(TensorDataset(torch.randn(7, 1, 28, 28), torch.arange(7)), batch_size=4)
        for cls in [MLP, CNN]:
            model = cls()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            run_epoch(model, loader, 'cpu', torch.optim.Adam(model.parameters(), lr=.001))
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
            trained = {k: v.clone() for k, v in model.state_dict().items()}
            result = run_epoch(model, loader, 'cpu')
            self.assertEqual(result['samples'], 7)
            self.assertTrue(np.isfinite(result['loss']))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(trained[key], value, rtol=0, atol=0)

    def test_loss_weighted_by_samples_not_batches(self):
        scores = torch.tensor([[5., 0.], [0., 5.], [5., 0.]])
        labels = torch.tensor([0, 1, 1])
        loader = DataLoader(TensorDataset(scores, labels), batch_size=2)
        result = run_epoch(nn.Identity(), loader, 'cpu')
        self.assertAlmostEqual(result['loss'], nn.CrossEntropyLoss()(scores, labels).item(), places=6)
        self.assertAlmostEqual(result['accuracy'], 2/3)

    def test_prediction_order_and_probabilities(self):
        labels = torch.tensor([9, 2, 1, 7, 0])
        loader = DataLoader(TensorDataset(torch.randn(5, 1, 28, 28), labels), batch_size=3)
        true, probabilities = predict(MLP(), loader, 'cpu')
        np.testing.assert_array_equal(true, labels.numpy())
        self.assertEqual(probabilities.shape, (5, 10))
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)

    def test_evaluation_artifacts_agree_on_synthetic_data(self):
        # 10k tensor giả: chỉ kiểm tra cơ chế xuất file, không phải kết quả Fashion-MNIST.
        test_root = Path(__file__).resolve().parents[1] / 'artifacts' / 'technical_tests'
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as directory:
            run = Path(directory)
            assert run.resolve().is_relative_to(test_root.resolve())
            for folder in ['checkpoints', 'metrics', 'figures']:
                (run / folder).mkdir()
            np.savez(run / 'splits.npz', train=[0], validation=[1])
            torch.save({'state_dict': MLP().state_dict(), 'classes': [str(k) for k in range(10)]},
                       run / 'checkpoints' / 'MLP_best.pt')
            loader = DataLoader(TensorDataset(torch.zeros(10000, 1, 28, 28),
                                             torch.arange(10000) % 10), batch_size=128)
            evaluate_models(loader, {'models': ['MLP']}, 'cpu', run, {'MLP': {}})
            import pandas as pd
            predictions = pd.read_csv(run / 'metrics' / 'MLP_predictions.csv')
            summary = pd.read_csv(run / 'metrics' / 'summary.csv')
            matrix = np.load(run / 'metrics' / 'MLP_confusion.npy')
            accuracy = (predictions.true_label == predictions.predicted_label).mean()
            self.assertEqual(len(predictions), 10000)
            self.assertAlmostEqual(accuracy, summary.accuracy.iloc[0])
            self.assertAlmostEqual(accuracy, matrix.trace() / matrix.sum())
            protocol = json.loads((run / 'frozen_protocol.json').read_text(encoding='utf-8'))
            self.assertEqual(predictions.checkpoint_hash.iloc[0], protocol['checkpoint_hashes']['MLP'])


if __name__ == '__main__':
    unittest.main()
