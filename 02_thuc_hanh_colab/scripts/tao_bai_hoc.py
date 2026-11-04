"""Tạo notebook mới; không mở, sửa hoặc chạy notebook cũ."""

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=text.strip() + "\n"))


def lesson(title, explanation, code, observe):
    md("## " + title + "\n\n" + explanation)
    compile(code, title, "exec")
    cells.append(
        dict(
            cell_type="code",
            metadata={},
            source=code.strip() + "\n",
            execution_count=None,
            outputs=[],
        )
    )
    md("**Quan sát / tự kiểm tra:** " + observe)


md(
    """# Fashion-MNIST — Học chậm Python, MLP và CNN

Đọc một mục, dự đoán đầu ra, rồi chạy ô kế tiếp.

1. Dữ liệu và batch (1–12)
2. Mô hình và bước học (13–23)
3. Huấn luyện, đánh giá và lưu (24–33)

Thực hiện huấn luyện trên Colab GPU."""
)
lesson(
    """1. Biến, danh sách và chỉ số trong Python""",
    """Dùng chỉ số để lấy một phần tử trong danh sách.""",
    """ten_lop_vi_du = ["áo", "quần", "túi"]
vi_tri = 2
print(ten_lop_vi_du[vi_tri])
print(len(ten_lop_vi_du))
print(ten_lop_vi_du[:2])""",
    """Kết quả: túi, 3, rồi danh sách áo/quần.""",
)
lesson(
    """2. Nạp công cụ làm việc với ảnh""",
    """Nạp thư viện cho ảnh, tensor và biểu đồ.""",
    """import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

print("PyTorch:", torch.__version__)""",
    """Nếu import lỗi, dừng ở đây để sửa môi trường; không chạy ô phụ thuộc.""",
)
lesson(
    """3. Cấu hình và thư mục riêng""",
    """Mỗi lần chạy lưu kết quả ở một thư mục riêng.""",
    """from uuid import uuid4

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.001
ROOT = Path("/content/fashion_mnist_hoc_cham")
DATA = ROOT / "data"
RUN = ROOT / "runs" / uuid4().hex
RUN.mkdir(parents=True, exist_ok=False)
print("Kết quả sẽ lưu tại:", RUN)""",
    """Chạy lại ô này tạo run mới: phải chạy lại các bước phụ thuộc để kết quả thuộc cùng run.""",
)
lesson(
    """4. Chọn thiết bị và đặt seed""",
    """Chọn thiết bị và đặt seed để lặp lại thí nghiệm.""",
    """import random

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_all(SEED)
print("Thiết bị:", DEVICE)""",
    """Seed giúp lặp lại trong cùng điều kiện, không bảo đảm giống bit trên mọi thiết bị/phiên bản.""",
)
lesson(
    """5. Tải train gốc và đọc một mẫu""",
    """Mỗi mẫu gồm một ảnh và một nhãn từ 0 đến 9.""",
    """raw_train = datasets.FashionMNIST(root=DATA, train=True, download=True)
CLASSES = raw_train.classes
image, label = raw_train[0]
print("Số mẫu:", len(raw_train))
print("Nhãn:", label, "Tên:", CLASSES[label])
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()""",
    """Chỉ số mẫu 0 không có nghĩa nhãn 0.""",
)
lesson(
    """6. Pixel, tensor, shape và dtype""",
    """Một ảnh xám có kích thước [1, 28, 28]. ToTensor đưa pixel về thang 0–1.""",
    """to_tensor = transforms.ToTensor()
pixel_tensor = to_tensor(image)
print(pixel_tensor.shape)
print(pixel_tensor.dtype)
print(pixel_tensor.min().item(), pixel_tensor.max().item())
print(pixel_tensor[0, 10:15, 10:15])""",
    """Dấu chấm không có () như `.shape` đọc thuộc tính; có () như `.min()` gọi phương thức.""",
)
lesson(
    """7. Chia 54k train và 6k validation""",
    """Chia ID của 60.000 ảnh theo tỷ lệ lớp.""",
    """indices = np.arange(len(raw_train))
all_labels = raw_train.targets.numpy()
train_idx, val_idx = train_test_split(
    indices, test_size=6000, random_state=SEED, stratify=all_labels
)
print(len(train_idx), len(val_idx))""",
    """Hãy giải thích vì sao hàm có chữ test nhưng ở đây tạo validation.""",
)
lesson(
    """8. Kiểm tra cách chia""",
    """Kiểm tra số ảnh từng lớp và giao nhau giữa hai tập.""",
    """train_set = set(train_idx.tolist())
val_set = set(val_idx.tolist())
assert len(train_set) == 54000
assert len(val_set) == 6000
assert len(train_set.intersection(val_set)) == 0
assert train_set.union(val_set) == set(indices.tolist())
for class_id in range(10):
    train_count = (all_labels[train_idx] == class_id).sum()
    val_count = (all_labels[val_idx] == class_id).sum()
    print(CLASSES[class_id], train_count, val_count)""",
    """`==` trên mảng cho một mảng True/False; sum đếm số True.""",
)
lesson(
    """9. Mean và standard deviation""",
    """Tính độ sáng trung bình và độ lệch chuẩn chỉ từ ảnh train.""",
    """train_pixels = raw_train.data[train_idx].float() / 255.0
MEAN = train_pixels.mean().item()
STD = train_pixels.std(unbiased=False).item()
assert STD > 0
del train_pixels
print("Mean:", MEAN, "Std:", STD)""",
    """Mean/std không phải tham số trọng số được optimizer học.""",
)
lesson(
    """10. Normalize ảnh và tạo các tập con""",
    """Dùng cùng mean và std khi chuẩn hóa cả ba tập.""",
    """transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
)
source = datasets.FashionMNIST(root=DATA, train=True, transform=transform)
train_ds = Subset(source, train_idx.tolist())
val_ds = Subset(source, val_idx.tolist())
test_ds = datasets.FashionMNIST(
    root=DATA, train=False, download=True, transform=transform
)
assert len(test_ds) == 10000
normalized_image, normalized_label = train_ds[0]
print(normalized_image.shape, normalized_label)""",
    """Pixel thay đổi vì tiền xử lý; optimizer không sửa pixel.""",
)
lesson(
    """11. Tạo DataLoader""",
    """DataLoader gom ảnh thành batch và xáo trộn tập train.""",
    """def make_train_loader():
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    return loader


val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)""",
    """Tạo loader không bắt đầu huấn luyện.""",
)
lesson(
    """12. Lấy một batch — điểm dừng buổi đầu""",
    """Đọc kích thước ảnh và nhãn trước khi đưa vào mô hình.""",
    """train_loader = make_train_loader()
batch_iterator = iter(train_loader)
images, labels = next(batch_iterator)
print("Ảnh:", images.shape, "Nhãn:", labels.shape)
print("Số batch:", len(train_loader))
view_image = images[0, 0] * STD + MEAN
plt.imshow(view_image.clamp(0, 1), cmap="gray")
plt.title(CLASSES[labels[0].item()])
plt.show()""",
    """DỪNG ở đây nếu mới học.""",
)
lesson(
    """13. Trọng số là hệ số nhân""",
    """Một pixel nhân với một trọng số. Quan sát trọng số trước và sau bước cập nhật.""",
    """x_demo = torch.tensor(0.8)
w_demo = torch.tensor(0.5, requires_grad=True)
y_demo = x_demo * w_demo
loss_demo = (y_demo - 1.0) ** 2
loss_demo.backward()
print("Dự đoán:", y_demo.item(), "Loss:", loss_demo.item())
print("Gradient:", w_demo.grad.item())
with torch.no_grad():
    w_demo -= 0.1 * w_demo.grad
print("Trọng số mới:", w_demo.item())""",
    """Gradient âm làm SGD cơ bản tăng trọng số.""",
)
lesson(
    """14. Định nghĩa MLP bằng class""",
    """MLP trải ảnh thành 784 số rồi tính 10 điểm lớp.""",
    """class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)


mlp_probe = MLP()
print(mlp_probe(images[:2]).shape)""",
    """128 trong Linear là số nút/đặc trưng, không phải batch size.""",
)
lesson(
    """15. Định nghĩa CNN""",
    """CNN dùng bộ lọc trên các vùng ảnh rồi phân loại từ đặc trưng đã học.""",
    """class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(1568, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits""",
    """CNN cũng Flatten, nhưng ở gần cuối sau khi học đặc trưng.""",
)
lesson(
    """16. Theo dõi shape qua từng lớp""",
    """In kích thước đầu ra sau từng lớp CNN.""",
    """cnn_probe = CNN()
probe_image = images[:1]
layers = list(cnn_probe.features) + list(cnn_probe.classifier)
with torch.no_grad():
    for layer in layers:
        probe_image = layer(probe_image)
        print(type(layer).__name__, tuple(probe_image.shape))""",
    """Sau Pool lần hai: [1,32,7,7]; sau Flatten: [1,1568]; cuối: [1,10].""",
)
lesson(
    """17. Chuẩn bị mô hình cho một bước học""",
    """Tạo MLP riêng để quan sát một lần cập nhật trên 16 ảnh.""",
    """seed_all(SEED)
demo_model = MLP().to(DEVICE)
demo_optimizer = torch.optim.Adam(demo_model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
xb = images[:16].to(DEVICE)
yb = labels[:16].to(DEVICE)
before = demo_model.layers[1].weight.detach().clone()""",
    """detach tách tensor khỏi đồ thị đạo hàm; clone sao chép dữ liệu để cập nhật không làm bản chụp thay đổi.""",
)
lesson(
    """18. Forward và logits""",
    """Mỗi hàng logits chứa 10 điểm. Vị trí có điểm lớn nhất là lớp dự đoán.""",
    """demo_optimizer.zero_grad()
logits = demo_model(xb)
probabilities = logits.softmax(dim=1)
predicted_labels = logits.argmax(dim=1)
print("Logits ảnh đầu:", logits[0].detach().cpu())
print("Tổng xác suất:", probabilities[0].sum().item())
print("Đoán / thật:", predicted_labels[0].item(), yb[0].item())""",
    """Logits không cần nằm trong 0–1.""",
)
lesson(
    """19. Loss đo sai lệch""",
    """CrossEntropyLoss nhận logits và nhãn nguyên.""",
    """loss = criterion(logits, yb)
print("Loss:", loss.item())""",
    """Tại đây chưa thay đổi trọng số.""",
)
lesson(
    """20. Backward tính gradient""",
    """Backward tính gradient của loss theo trọng số.""",
    """loss.backward()
gradient = demo_model.layers[1].weight.grad
print("Gradient của 5 trọng số:", gradient[0, :5].detach().cpu())
print("Trọng số còn nguyên:", torch.equal(before, demo_model.layers[1].weight.detach()))""",
    """Dấu gradient chỉ hướng tăng loss cho SGD cơ bản.""",
)
lesson(
    """21. Step cập nhật trọng số""",
    """Adam dùng gradient để cập nhật trọng số.""",
    """demo_optimizer.step()
after = demo_model.layers[1].weight.detach()
change = (after - before).abs().mean().item()
with torch.no_grad():
    new_loss = criterion(demo_model(xb), yb).item()
print("Thay đổi trọng số:", change)
print("Loss trước / sau:", loss.item(), new_loss)
assert change > 0""",
    """Có thể dừng buổi thứ hai ở đây.""",
)
lesson(
    """22. Hàm train một epoch""",
    """Một epoch gồm các bước học trên toàn bộ batch train.""",
    """def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_images = 0
    for batch_images, batch_labels in loader:
        batch_images = batch_images.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        optimizer.zero_grad()
        batch_logits = model(batch_images)
        batch_loss = criterion(batch_logits, batch_labels)
        batch_loss.backward()
        optimizer.step()
        batch_count = batch_labels.size(0)
        predicted = batch_logits.argmax(dim=1)
        total_correct += (predicted == batch_labels).sum().item()
        total_loss += batch_loss.item() * batch_count
        total_images += batch_count
    return total_loss / total_images, total_correct / total_images""",
    """Ô định nghĩa hàm chưa train.""",
)
lesson(
    """23. Hàm validation không cập nhật""",
    """Đo loss và accuracy trên validation với trọng số cố định.""",
    """def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            batch_logits = model(batch_images)
            batch_loss = criterion(batch_logits, batch_labels)
            batch_count = batch_labels.size(0)
            predicted = batch_logits.argmax(dim=1)
            total_correct += (predicted == batch_labels).sum().item()
            total_loss += batch_loss.item() * batch_count
            total_images += batch_count
    return total_loss / total_images, total_correct / total_images""",
    """So sánh với bước 22: không có zero_grad/backward/step.""",
)
lesson(
    """24. Chuẩn bị lưu kết quả — phần kỹ thuật phụ trợ""",
    """Lưu cấu hình, ID của split và thống kê chuẩn hóa.""",
    """import json
import pandas as pd
import time
import hashlib


def save_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


MODEL_CLASSES = {"MLP": MLP, "CNN": CNN}
CONFIG = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LR,
    "device": str(DEVICE),
    "torch": str(torch.__version__),
}
save_json(RUN / "config.json", CONFIG)
np.savez(RUN / "splits.npz", train=train_idx, validation=val_idx)
save_json(RUN / "preprocessing.json", {"mean": MEAN, "std": STD, "classes": CLASSES})""",
    """Kết quả mới nằm trong RUN riêng.""",
)
lesson(
    """25. Lặp các epoch và chọn checkpoint""",
    """Sau mỗi epoch, lưu trọng số nếu validation loss giảm.""",
    """def fit(name):
    seed_all(SEED)
    model = MODEL_CLASSES[name]().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = make_train_loader()
    best_loss = float("inf")
    history = []
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        }
        history.append(row)
        if val_loss < best_loss:
            best_loss = val_loss
            payload = {
                "state_dict": model.state_dict(),
                "model_name": name,
                "epoch": epoch,
                "mean": MEAN,
                "std": STD,
                "classes": CLASSES,
                "config": CONFIG,
            }
            torch.save(payload, RUN / f"{name}_best.pt")
        pd.DataFrame(history).to_csv(RUN / f"{name}_history.csv", index=False)
        print(
            f"{name} epoch {epoch}: train={train_loss:.4f}, validation={val_loss:.4f}",
            flush=True,
        )
    elapsed = time.perf_counter() - start
    parameter_count = sum(p.numel() for p in model.parameters())
    info = {
        "seconds": elapsed,
        "parameters": parameter_count,
        "best_val_loss": best_loss,
    }
    save_json(RUN / f"{name}_training.json", info)
    return info""",
    """sum(p.numel() for p in ...) là cộng số phần tử của từng tensor tham số.""",
)
lesson(
    """26. Huấn luyện MLP — chỉ chạy khi đã hiểu""",
    """Train MLP trên Colab GPU với cấu hình đã chọn.""",
    """training_info = {}
training_info["MLP"] = fit("MLP")""",
    """Kỳ vọng in 5 dòng epoch.""",
)
lesson(
    """27. Huấn luyện CNN và nhìn đường cong""",
    """Train CNN cùng cấu hình rồi vẽ hai đường loss.""",
    """training_info["CNN"] = fit("CNN")
for name in ["MLP", "CNN"]:
    history = pd.read_csv(RUN / f"{name}_history.csv")
    plt.figure(figsize=(6, 3))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN / f"{name}_learning_curve.png")
    plt.show()""",
    """Điểm validation tốt hơn train chưa tự chứng minh không overfit: train được đo trong lúc trọng số thay đổi.""",
)
lesson(
    """28. Chốt model trước khi xem test""",
    """Nạp checkpoint có validation loss thấp nhất trước khi xem test.""",
    """checkpoint_hashes = {}
trained_models = {}
for name in ["MLP", "CNN"]:
    path = RUN / f"{name}_best.pt"
    checkpoint_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(path, map_location=DEVICE, weights_only=True)
    model = MODEL_CLASSES[name]().to(DEVICE)
    model.load_state_dict(payload["state_dict"])
    trained_models[name] = model.eval()
save_json(
    RUN / "frozen_protocol.json",
    {
        "selection": "minimum validation loss; earliest tie",
        "checkpoint_hashes": checkpoint_hashes,
        "split_hash": hashlib.sha256((RUN / "splits.npz").read_bytes()).hexdigest(),
    },
)
print("Đã chốt checkpoint; không sửa cấu hình theo test.")""",
    """Chỉ dùng checkpoint tự tạo/tin cậy.""",
)
lesson(
    """29. Thu thập dự đoán test""",
    """Thu nhãn thật và xác suất theo thứ tự ảnh test.""",
    """def predict_test(model):
    all_labels = []
    all_probabilities = []
    model.eval()
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(DEVICE)
            batch_logits = model(batch_images)
            batch_probs = batch_logits.softmax(dim=1)
            all_probabilities.append(batch_probs.cpu().numpy())
            all_labels.append(batch_labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probabilities)""",
    """Softmax cao không đảm bảo đúng.""",
)
lesson(
    """30. So sánh MLP/CNN bằng một nguồn số liệu""",
    """Tính accuracy và macro-F1 từ cùng bảng dự đoán.""",
    """from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

predictions = {}
rows = []
for name in ["MLP", "CNN"]:
    y_true, probs = predict_test(trained_models[name])
    y_pred = probs.argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    assert len(y_true) == 10000 and matrix.sum() == 10000
    assert np.isclose(np.trace(matrix) / 10000, accuracy)
    table = pd.DataFrame(
        {
            "sample_id": np.arange(10000),
            "true_label": y_true,
            "predicted_label": y_pred,
            "checkpoint_hash": checkpoint_hashes[name],
        }
    )
    for class_id in range(10):
        table[f"prob_{class_id}"] = probs[:, class_id]
    table.to_csv(RUN / f"{name}_predictions.csv", index=False)
    predictions[name] = {
        "true": y_true,
        "pred": y_pred,
        "probs": probs,
        "matrix": matrix,
    }
    rows.append(
        {
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "parameters": training_info[name]["parameters"],
        }
    )
summary = pd.DataFrame(rows)
summary.to_csv(RUN / "summary.csv", index=False)
print(summary.to_string(index=False))
print("EVALUATION_COMPLETE")""",
    """Các số liệu chỉ có sau khi chạy thật.""",
)
lesson(
    """31. Xem ma trận nhầm lẫn và một ảnh sai""",
    """Ma trận có hàng là nhãn thật và cột là nhãn dự đoán.""",
    """cnn_result = predictions["CNN"]
plt.figure(figsize=(8, 7))
plt.imshow(cnn_result["matrix"], cmap="Blues")
plt.xticks(range(10), CLASSES, rotation=90)
plt.yticks(range(10), CLASSES)
plt.xlabel("Dự đoán")
plt.ylabel("Nhãn thật")
plt.colorbar()
plt.tight_layout()
plt.savefig(RUN / "CNN_confusion.png")
plt.show()
wrong_ids = np.flatnonzero(cnn_result["true"] != cnn_result["pred"])
if len(wrong_ids) > 0:
    sample_id = int(wrong_ids[0])
    wrong_image, true_label = test_ds[sample_id]
    predicted_label = int(cnn_result["pred"][sample_id])
    plt.imshow(wrong_image[0] * STD + MEAN, cmap="gray")
    plt.title(f"Thật: {CLASSES[true_label]} / Đoán: {CLASSES[predicted_label]}")
    plt.show()""",
    """Hàng Shirt, cột T-shirt/top: áo sơ mi bị đoán thành áo thun.""",
)
lesson(
    """32. Nhìn đặc trưng CNN và dự đoán cùng ảnh""",
    """Xem phản ứng của bốn bộ lọc đầu tiên trên cùng một ảnh.""",
    """sample_id = 0
sample_image, sample_label = test_ds[sample_id]
one_image_batch = sample_image.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    feature_maps = trained_models["CNN"].features[:2](one_image_batch)
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
axes[0].imshow(sample_image[0] * STD + MEAN, cmap="gray")
axes[0].set_title(CLASSES[sample_label])
for channel in range(4):
    axes[channel + 1].imshow(feature_maps[0, channel].cpu(), cmap="viridis")
    axes[channel + 1].set_title(f"Kênh {channel}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(RUN / "CNN_feature_maps.png")
plt.show()
for name in ["MLP", "CNN"]:
    predicted_label = int(predictions[name]["pred"][sample_id])
    print(name, "dự đoán:", CLASSES[predicted_label], "| thật:", CLASSES[sample_label])""",
    """Không yêu cầu hai mô hình đoán giống nhau.""",
)
lesson(
    """33. Lưu kết quả và tải về""",
    """Đóng gói trọng số và kết quả để tải về trước khi kết thúc phiên.""",
    """import shutil
from google.colab import files

save_json(
    RUN / "STATUS.json",
    {
        "stage": "completed_learning_notebook",
        "epochs": EPOCHS,
        "test_samples": 10000,
        "limits": ["one seed", "five epochs", "no ablation", "no exact resume"],
    },
)
archive_path = shutil.make_archive(str(RUN), "zip", root_dir=RUN)
print("Kết quả:", archive_path)
files.download(archive_path)""",
    """Nếu tải bị chặn, mở Files bên trái, tìm file ZIP theo đường dẫn vừa in rồi Download.""",
)
md(
    """## Sẵn sàng cho giai đoạn 3 khi nào?

Tự kiểm tra: với batch 64 ảnh, hãy dự đoán shape của ảnh, nhãn và logits.

Đáp án: [64,1,28,28], [64] và [64,10].

[Dữ liệu Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)"""
)

for i, cell in enumerate(cells):
    cell["id"] = f"hoc-cham-{i:03d}"
    cell["metadata"]["id"] = cell["id"]
notebook = dict(
    nbformat=4,
    nbformat_minor=5,
    metadata={
        "colab": {
            "name": "01_hoc_tung_buoc.ipynb",
            "provenance": [],
        },
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    cells=cells,
)
target = NOTEBOOKS / "01_hoc_tung_buoc.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(target)
print("Cells:", len(cells), "Code:", sum(c["cell_type"] == "code" for c in cells))
