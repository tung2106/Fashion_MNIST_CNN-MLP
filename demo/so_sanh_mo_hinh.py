"""Tạo hình trình chiếu từ dự đoán CPU của ba checkpoint Fashion-MNIST."""

import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt
from PIL import Image

from models.du_doan import load_model, predict_image
from utils.duong_dan import DEFAULT_CHECKPOINT, ROOT

EXAMPLES = ROOT / "demo" / "examples"
OUTPUT = ROOT / "demo" / "output"
CHECKPOINTS = {
    "MLP": DEFAULT_CHECKPOINT.with_name("mlp.pt"),
    "CNN cơ bản": DEFAULT_CHECKPOINT.with_name("cnn.pt"),
    "CNNPlus": DEFAULT_CHECKPOINT,
}


def render_sample(sample_id):
    samples = json.loads((EXAMPLES / "labels.json").read_text(encoding="utf-8"))
    labels_by_id = {row["sample_id"]: row["true_label"] for row in samples}
    if sample_id not in labels_by_id:
        available = ", ".join(map(str, labels_by_id))
        raise ValueError(f"Không có ảnh ID {sample_id}. ID có sẵn: {available}.")

    missing = [path.name for path in CHECKPOINTS.values() if not path.is_file()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Thiếu checkpoint trong models/checkpoints: {names}.")

    with Image.open(EXAMPLES / f"sample_{sample_id}.png") as source:
        image = source.copy()
    true_label = labels_by_id[sample_id]

    comparison = []
    for name, checkpoint in CHECKPOINTS.items():
        model, payload, transform = load_model(checkpoint)
        scores = predict_image(model, image, transform)
        classes = payload["classes"]
        predicted_label = scores.argmax().item()
        verdict = "Đúng" if predicted_label == true_label else "Sai"
        comparison.append(
            (name, classes[predicted_label], f"{scores[predicted_label]:.4%}", verdict)
        )
        if name == "CNNPlus":
            top_scores, top_labels = scores.topk(3)
            top_names = [classes[label] for label in top_labels.tolist()]
            true_name = classes[true_label]

    fig, axes = plt.subplot_mosaic(
        [["image", "scores"], ["comparison", "comparison"]],
        figsize=(10, 6.1),
        height_ratios=(1.4, 1),
        layout="constrained",
    )
    fig.suptitle(f"Fashion-MNIST · Ảnh test #{sample_id}", fontsize=20, weight="bold")
    axes["image"].imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    axes["image"].set_title(f"Nhãn thật · {true_name}", fontsize=14)
    axes["image"].axis("off")

    scores_ax = axes["scores"]
    scores_ax.set_title("Top 3 của CNNPlus", fontsize=14, loc="left")
    scores_ax.barh(top_names, top_scores.tolist(), color="#2563a4")
    scores_ax.invert_yaxis()
    scores_ax.set_xlim(0, 1.27)
    scores_ax.set_xticks([])
    for index, score in enumerate(top_scores.tolist()):
        scores_ax.text(score + 0.02, index, f"{score:.4%}", va="center", fontsize=11)
    scores_ax.spines[:].set_visible(False)
    scores_ax.tick_params(axis="y", length=0, labelsize=11)

    table_ax = axes["comparison"]
    table_ax.axis("off")
    table = table_ax.table(
        cellText=comparison,
        colLabels=("Mô hình", "Dự đoán", "Điểm softmax", "So với nhãn thật"),
        cellLoc="center",
        bbox=(0.02, 0.2, 0.96, 0.73),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#e6eef7" if row == 0 else "#f1f5f9")
        if row == 0:
            cell.set_text_props(weight="bold")
        elif col == 3:
            cell.set_text_props(
                color="#166534" if comparison[row - 1][3] == "Đúng" else "#9f1239"
            )
    table_ax.text(
        0.02, 0.04, "Điểm softmax cao vẫn có thể đi kèm dự đoán sai.", fontsize=11
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=19, help="ID ảnh trong tập test")
    parser.add_argument("--output", type=Path, help="Đường dẫn lưu hình PNG")
    args = parser.parse_args()

    plt.switch_backend("Agg")
    try:
        figure = render_sample(args.sample)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    output = args.output or OUTPUT / f"sample_{args.sample}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="png", dpi=150)
    plt.close(figure)
    print(output.resolve())


if __name__ == "__main__":
    main()
