"""Dự đoán một ảnh xám 28x28 bằng checkpoint, không train."""

import argparse
from pathlib import Path

from PIL import Image

from models.du_doan import load_model, predict_image
from utils.duong_dan import DEFAULT_CHECKPOINT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    model, payload, transform = load_model(args.checkpoint)
    with Image.open(args.image) as image:
        probs = predict_image(model, image, transform)

    for label in probs.argsort(descending=True)[:3]:
        print(f"{payload['classes'][label]}: {probs[label]:.2%}")

    print("Điểm softmax không phải độ tin cậy đã được hiệu chuẩn.")


if __name__ == "__main__":
    main()
