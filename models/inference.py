"""Nạp checkpoint và dự đoán; không có bước huấn luyện."""

import torch
from torchvision import transforms

from models.networks import MODEL_CLASSES


def load_model(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = MODEL_CLASSES[payload["model_name"]]().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((payload["mean"],), (payload["std"],)),
        ]
    )
    return model, payload, transform


def predict_image(model, image, transform, device="cpu"):
    if image.size != (28, 28) or image.mode != "L":
        raise ValueError("Demo cơ bản nhận ảnh xám 28x28 theo Fashion-MNIST.")
    with torch.inference_mode():
        logits = model(transform(image).unsqueeze(0).to(device))
    return logits.softmax(1)[0].cpu()
