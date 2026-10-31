"""Hình luôn được vẽ từ dữ liệu của chính run đang chạy."""
import matplotlib
matplotlib.use('Agg')  # Lưu PNG được cả trên máy không có cửa sổ đồ họa.
import matplotlib.pyplot as plt
import pandas as pd
import torch


def plot_samples(data, run):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        image, label = data['train'][i]
        ax.imshow(image[0] * data['std'] + data['mean'], cmap='gray', vmin=0, vmax=1)
        ax.set_title(data['classes'][label])
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(run / 'figures' / 'train_samples.png')
    plt.close(fig)


def plot_history(name, run):
    history = pd.read_csv(run / 'metrics' / f'{name}_history.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for split in ['train', 'val']:
            ax.plot(history['epoch'], history[f'{split}_{metric}'], marker='o', label=split)
        ax.set(xlabel='Epoch', ylabel=metric, title=name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(run / 'figures' / f'{name}_learning_curve.png')
    plt.close(fig)


def plot_matrix(matrix, classes, name, run):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(10), yticks=range(10), xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label', ylabel='True label', title=name)
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center',
                    color='white' if matrix[row, col] > matrix.max()/2 else 'black')
    fig.tight_layout()
    fig.savefig(run / 'figures' / f'{name}_confusion.png')
    plt.close(fig)


def plot_features(model, dataset, payload, device, run):
    image, label = dataset[0]  # ID 0 của tập test, ghi trong metadata kèm hình.
    model.eval()
    with torch.no_grad():
        features = model.features[:2](image.unsqueeze(0).to(device))[0].cpu()
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    axes[0].imshow(image[0] * payload['std'] + payload['mean'], cmap='gray')
    axes[0].set_title(payload['classes'][label])
    for k in range(4):
        axes[k+1].imshow(features[k], cmap='viridis')
        axes[k+1].set_title(f'Channel {k}')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(run / 'figures' / 'CNN_feature_maps.png')
    plt.close(fig)
