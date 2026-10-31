"""Nạp best checkpoint, lưu dự đoán từng mẫu, rồi tính chỉ số từ chúng."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from .models import MODEL_CLASSES
from .plots import plot_matrix, plot_features
from .utils import save_json, file_hash


def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for images, targets in loader:
            probabilities.append(model(images.to(device)).softmax(1).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def evaluate_models(loader, config, device, run, training_info):
    hashes = {name: file_hash(run / 'checkpoints' / f'{name}_best.pt') for name in config['models']}
    save_json(run / 'frozen_protocol.json', {
        'selection': 'minimum validation loss; earliest tie', 'checkpoint_hashes': hashes,
        'split_hash': file_hash(run / 'splits.npz'),
        'note': 'Test đã được xem ở giai đoạn 2; đây không phải holdout chưa từng biết.'})
    rows = []
    for name in config['models']:
        payload = torch.load(run / 'checkpoints' / f'{name}_best.pt',
                             map_location=device, weights_only=True)
        model = MODEL_CLASSES[name]().to(device)
        model.load_state_dict(payload['state_dict'])
        true, probs = predict(model, loader, device)
        pred = probs.argmax(1)
        matrix = confusion_matrix(true, pred, labels=range(10))
        accuracy = accuracy_score(true, pred)
        assert len(true) == 10000 and matrix.sum() == len(true)
        assert np.isclose(matrix.trace() / len(true), accuracy)
        table = pd.DataFrame(dict(sample_id=np.arange(len(true)), split='test', run_id=run.name,
                                  true_label=true, predicted_label=pred, checkpoint_hash=hashes[name]))
        for k in range(10):
            table[f'prob_{k}'] = probs[:, k]
        table.to_csv(run / 'metrics' / f'{name}_predictions.csv', index=False)
        np.save(run / 'metrics' / f'{name}_confusion.npy', matrix)
        save_json(run / 'metrics' / f'{name}_classification_report.json',
                  classification_report(true, pred, labels=list(range(10)),
                                        target_names=payload['classes'], output_dict=True, zero_division=0))
        rows.append(dict(model=name, accuracy=accuracy,
                         macro_f1=f1_score(true, pred, labels=list(range(10)), average='macro', zero_division=0),
                         **training_info[name]))
        plot_matrix(matrix, payload['classes'], name, run)
        if name == 'CNN':
            plot_features(model, loader.dataset, payload, device, run)
            save_json(run / 'figures' / 'feature_metadata.json', {
                'sample_id': 0, 'split': 'test', 'layer': 'features[:2] (Conv + ReLU)',
                'checkpoint_hash': hashes[name], 'channels': [0, 1, 2, 3],
                'display': 'independent color scale per channel; descriptive, not causal'})
    summary = pd.DataFrame(rows)
    summary.to_csv(run / 'metrics' / 'summary.csv', index=False)
    print(summary.to_string(index=False))
