"""Tải dữ liệu, chia tập, chuẩn hóa và gom batch. Không học trọng số ở đây."""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from .utils import save_json


def split_indices(labels, seed):
    # Chia ID ảnh, stratify giữ mỗi lớp có cùng tỷ lệ giữa hai tập.
    train, val = train_test_split(np.arange(len(labels)), test_size=6000,
                                  random_state=seed, stratify=np.asarray(labels))
    assert len(train) == 54000 and len(val) == 6000
    assert not np.intersect1d(train, val).size
    return train, val


def prepare_data(root, config, run):
    raw = datasets.FashionMNIST(root, train=True, download=True)
    train_ids, val_ids = split_indices(raw.targets, config['seed'])
    # Smoke dùng ít ảnh mỗi lớp, giữ nguyên split gốc để không lẫn train/val.
    if config['profile'] == 'smoke':
        def take_per_class(ids, n):
            return np.concatenate([ids[raw.targets.numpy()[ids] == k][:n] for k in range(10)])
        train_ids = take_per_class(train_ids, 32)
        val_ids = take_per_class(val_ids, 8)
    pixels = raw.data[train_ids].float() / 255.0
    mean, std = pixels.mean().item(), pixels.std(unbiased=False).item()
    del pixels  # Không giữ mảng float tạm trong suốt quá trình train.
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))])
    source = datasets.FashionMNIST(root, train=True, transform=transform)
    data = {'train': Subset(source, train_ids.tolist()),
            'validation': Subset(source, val_ids.tolist()),
            'mean': mean, 'std': std, 'classes': raw.classes, 'transform': transform}
    np.savez(run / 'splits.npz', train=train_ids, validation=val_ids)
    save_json(run / 'preprocessing.json', {
        'mean': mean, 'std': std, 'computed_on': 'selected_train_only',
        'train_samples': len(train_ids), 'validation_samples': len(val_ids),
        'classes': raw.classes,
        'train_counts': np.bincount(raw.targets[train_ids], minlength=10).tolist(),
        'validation_counts': np.bincount(raw.targets[val_ids], minlength=10).tolist()})
    return data


def make_loader(dataset, config, training=False):
    # Mỗi model nhận loader mới với cùng seed: cùng quy tắc xáo trộn.
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=training,
                      generator=torch.Generator().manual_seed(config['seed']), num_workers=0)


def load_test(root, data):
    # Chỉ gọi khi người dùng yêu cầu đánh giá các checkpoint đã chốt.
    test = datasets.FashionMNIST(root, train=False, download=True, transform=data['transform'])
    assert len(test) == 10000
    return test
