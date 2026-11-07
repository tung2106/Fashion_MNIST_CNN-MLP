"""Ngăn huấn luyện nhầm trên máy cá nhân."""

import importlib.util

import torch


def is_colab_gpu():
    try:
        on_colab = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        on_colab = False
    return on_colab and torch.cuda.is_available()


def require_colab_gpu():
    if not is_colab_gpu():
        raise RuntimeError(
            "Huấn luyện và kiểm tra có optimizer chỉ chạy trên Google Colab GPU."
        )
