# %%
from pathlib import Path  # Tạo đường dẫn độc lập hệ điều hành.
from datetime import datetime, timezone  # Ghi thời điểm chạy.
import json, random, time, hashlib, uuid, shutil, platform  # Lưu cấu hình và truy vết.
import numpy as np  # Xử lý mảng và lưu predictions.
import pandas as pd  # Bảng kết quả.
import matplotlib.pyplot as plt  # Vẽ ảnh và biểu đồ.
import torch  # Tensor, đạo hàm tự động và huấn luyện.
from torch import nn  # Các lớp mạng và hàm mất mát.
from torch.utils.data import DataLoader, Subset  # Nạp batch và chọn tập con.
import torchvision
from torchvision import datasets, transforms  # Fashion-MNIST và xử lý ảnh.
from sklearn.model_selection import train_test_split  # Chia có giữ tỉ lệ lớp.
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from IPython.display import display  # Hiển thị bảng trong notebook.

SEED, BATCH_SIZE, EPOCHS, LR = 42, 128, 5, 0.001  # Ngân sách học tập ban đầu.
ROOT = Path('/content/fashion_mnist_basic')  # Máy tạm Colab.
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
RUN = ROOT / 'runs' / RUN_ID  # Mỗi lần chạy một thư mục riêng.
for folder in [ROOT/'data', RUN/'checkpoints', RUN/'metrics', RUN/'figures']:
    folder.mkdir(parents=True, exist_ok=True)  # Tạo nếu chưa có.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def seed_all(seed):
    random.seed(seed)  # Ngẫu nhiên của Python.
    np.random.seed(seed)  # Ngẫu nhiên của NumPy.
    torch.manual_seed(seed)  # Ngẫu nhiên của PyTorch.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Ngẫu nhiên trên GPU.
    torch.backends.cudnn.benchmark = False  # Tránh tự đổi thuật toán theo benchmark.
    torch.backends.cudnn.deterministic = True  # Ưu tiên thuật toán xác định khi hỗ trợ.

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

seed_all(SEED)
CONFIG = dict(seed=SEED, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
              profile='learning_full_data_5_epochs', device=str(DEVICE),
              python=platform.python_version(), torch=str(torch.__version__),
              torchvision=str(torchvision.__version__), classes=CLASSES)
save_json(RUN/'config.json', CONFIG)
print('Thiết bị:', DEVICE, '| GPU:', torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'Không có')
print('Run:', RUN, '| Epochs:', EPOCHS)


# %%
raw_train = datasets.FashionMNIST(ROOT/'data', train=True, download=True)  # 60k ảnh + nhãn.
indices = np.arange(len(raw_train))  # ID từ 0 đến 59999 trong train gốc.
train_idx, val_idx = train_test_split(indices, test_size=6000, random_state=SEED,
                                     stratify=raw_train.targets.numpy())
assert len(train_idx) == 54000 and len(val_idx) == 6000
assert not set(train_idx).intersection(val_idx)  # Không ảnh nào vừa train vừa validation.
np.savez(RUN/'splits.npz', train=train_idx, validation=val_idx)
train_pixels = raw_train.data[train_idx].float() / 255.0  # Chỉ đọc 54k train để tính thống kê.
MEAN = train_pixels.mean().item()  # Độ sáng trung bình.
STD = train_pixels.std(unbiased=False).item()  # Độ lệch chuẩn tổng thể.
del train_pixels  # Giải phóng mảng tạm.
save_json(RUN/'preprocessing.json', dict(mean=MEAN, std=STD, computed_on='train_54000'))
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
source = datasets.FashionMNIST(ROOT/'data', train=True, transform=transform)
train_ds = Subset(source, train_idx.tolist())  # Tập học trọng số.
val_ds = Subset(source, val_idx.tolist())  # Tập chọn checkpoint.
test_ds = datasets.FashionMNIST(ROOT/'data', train=False, download=True, transform=transform)
assert len(test_ds) == 10000

def make_train_loader():
    generator = torch.Generator().manual_seed(SEED)  # Thứ tự batch khởi đầu giống nhau cho hai model.
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
counts = pd.DataFrame({'class': CLASSES,
    'train': np.bincount(raw_train.targets[train_idx].numpy(), minlength=10),
    'validation': np.bincount(raw_train.targets[val_idx].numpy(), minlength=10)})
display(counts)
print('Train / validation / test:', len(train_ds), len(val_ds), len(test_ds))
print('Mean / std:', MEAN, STD)


# %%
images, labels = next(iter(make_train_loader()))  # Lấy một batch, chưa huấn luyện.
print('Ảnh:', images.shape, images.dtype, '| Nhãn:', labels.shape, labels.dtype)
print('Nhãn đầu:', labels[0].item(), CLASSES[labels[0].item()])
print('Vùng pixel 5x5 sau normalize:', images[0, 0, 10:15, 10:15])
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, img, label in zip(axes.flat, images[:10], labels[:10]):
    ax.imshow((img[0] * STD + MEAN).clamp(0, 1), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'{label.item()}: {CLASSES[label.item()]}')
    ax.axis('off')
plt.tight_layout()
fig.savefig(RUN/'figures'/'train_samples.png', dpi=140)
plt.show()


# %%
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # Khởi tạo khả năng quản lý tham số của nn.Module.
        self.layers = nn.Sequential(
            nn.Flatten(),  # [B,1,28,28] -> [B,784].
            nn.Linear(784, 128),  # Học 128 tổ hợp từ 784 pixel.
            nn.ReLU(),  # Thêm tính phi tuyến.
            nn.Linear(128, 10),  # [B,128] -> [B,10] logits.
        )
    def forward(self, x):
        return self.layers(x)  # Không Softmax trước CrossEntropyLoss.

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B,16,28,28].
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B,16,14,14].
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B,32,14,14].
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B,32,7,7].
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B,32*7*7] = [B,1568].
            nn.Linear(32*7*7, 128),  # Tổng hợp đặc trưng.
            nn.ReLU(),
            nn.Linear(128, 10),  # Điểm số 10 lớp.
        )
    def forward(self, x):
        return self.classifier(self.features(x))

MODEL_CLASSES = {'MLP': MLP, 'CNN': CNN}
for name, cls in MODEL_CLASSES.items():
    model = cls()  # Bản tạm để quan sát, chưa train.
    print(name, '| tham số:', sum(p.numel() for p in model.parameters()),
          '| output:', tuple(model(images[:2]).shape))
probe = CNN()
x = images[:1]  # Dùng một ảnh để theo dõi shape.
for layer in list(probe.features) + list(probe.classifier):
    x = layer(x)
    print(type(layer).__name__, '->', tuple(x.shape))


# %%
seed_all(SEED)
demo_model = MLP().to(DEVICE)  # Model minh họa riêng.
demo_optimizer = torch.optim.Adam(demo_model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()  # Nhận logits, không cần Softmax ngoài.
xb, yb = images[:16].to(DEVICE), labels[:16].to(DEVICE)
before = next(demo_model.parameters()).detach().clone()  # Chụp trọng số trước cập nhật.
demo_optimizer.zero_grad(set_to_none=True)
logits = demo_model(xb)  # Forward: [16,10].
loss = criterion(logits, yb)
loss.backward()  # Backpropagation: tạo .grad cho tham số.
gradient_norm = next(demo_model.parameters()).grad.norm().item()
demo_optimizer.step()  # Cập nhật trọng số.
change = (next(demo_model.parameters()).detach() - before).abs().mean().item()
with torch.no_grad():
    after_loss = criterion(demo_model(xb), yb).item()
print('Logits:', logits.shape, '| Loss trước/sau:', loss.item(), after_loss)
print('Độ lớn gradient:', gradient_norm, '| Mức thay đổi trọng số trung bình:', change)
assert gradient_norm > 0 and change > 0  # Xác nhận đã có bước học thật.
del demo_model, demo_optimizer, before  # Không dùng lại model minh họa.


# %%
def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None  # Có optimizer thì cập nhật trọng số.
    model.train(training)  # False tương đương eval().
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):  # Chỉ lưu gradient khi train.
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                loss.backward()
                optimizer.step()
            n = yb.size(0)  # Batch cuối có thể ít hơn BATCH_SIZE.
            total_loss += loss.item() * n
            correct += (logits.argmax(1) == yb).sum().item()
            count += n
    return {'loss': total_loss/count, 'accuracy': correct/count}

def fit(name):
    seed_all(SEED)  # Khởi tạo có thể lặp lại.
    model = MODEL_CLASSES[name]().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = make_train_loader()
    best_loss, best_epoch, history = float('inf'), 0, []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS+1):
        tick = time.perf_counter()
        train = run_epoch(model, loader, optimizer)
        val = run_epoch(model, val_loader)  # Không truyền optimizer.
        row = dict(epoch=epoch, train_loss=train['loss'], train_accuracy=train['accuracy'],
                   val_loss=val['loss'], val_accuracy=val['accuracy'], seconds=time.perf_counter()-tick)
        history.append(row)
        payload = dict(model_name=name, state_dict=model.state_dict(), epoch=epoch,
                       config=CONFIG, mean=MEAN, std=STD, classes=CLASSES)
        if val['loss'] < best_loss:
            best_loss, best_epoch = val['loss'], epoch
            torch.save(payload, RUN/'checkpoints'/f'{name}_best.pt')
        torch.save({**payload, 'optimizer': optimizer.state_dict(), 'history': history,
                    'best_loss': best_loss, 'best_epoch': best_epoch}, RUN/'checkpoints'/f'{name}_last.pt')
        pd.DataFrame(history).to_csv(RUN/'metrics'/f'{name}_history.csv', index=False)
        save_json(RUN/'STATUS.json', dict(stage='training', model=name, epoch=epoch, run_id=RUN_ID))
        print(f"{name} {epoch}/{EPOCHS} | train loss={train['loss']:.4f}, acc={train['accuracy']:.4f} | val loss={val['loss']:.4f}, acc={val['accuracy']:.4f} | {row['seconds']:.1f}s", flush=True)
    info = dict(best_epoch=best_epoch, best_val_loss=best_loss,
                train_seconds=time.perf_counter()-started,
                parameters=sum(p.numel() for p in model.parameters()))
    save_json(RUN/'metrics'/f'{name}_training.json', info)
    return info


# %%
training_info = {}  # Lưu thông tin hai model.
for name in ['MLP', 'CNN']:
    training_info[name] = fit(name)
print('TRAIN_COMPLETE', training_info)


# %%
for name in ['MLP', 'CNN']:
    h = pd.read_csv(RUN/'metrics'/f'{name}_history.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        ax.plot(h.epoch, h[f'train_{metric}'], label='train')
        ax.plot(h.epoch, h[f'val_{metric}'], label='validation')
        ax.set(xlabel='Epoch', ylabel=metric, title=f'{name}: {metric}')
        ax.legend()
    plt.tight_layout()
    fig.savefig(RUN/'figures'/f'{name}_curves.png', dpi=140)
    plt.show()

checkpoint_hashes = {name: hashlib.sha256((RUN/'checkpoints'/f'{name}_best.pt').read_bytes()).hexdigest()
                     for name in ['MLP', 'CNN']}
save_json(RUN/'frozen_protocol.json', dict(run_id=RUN_ID, selection='minimum validation loss; earliest tie',
    checkpoints=checkpoint_hashes, split_hash=hashlib.sha256((RUN/'splits.npz').read_bytes()).hexdigest(),
    frozen_at_utc=datetime.now(timezone.utc).isoformat(), training=training_info))
print('Đã chốt checkpoint trước test:', checkpoint_hashes)


# %%
@torch.no_grad()  # Không cần gradient khi đánh giá.
def collect_predictions(model):
    model.eval()
    all_y, all_prob = [], []
    for xb, yb in test_loader:
        probabilities = model(xb.to(DEVICE)).softmax(dim=1).cpu().numpy()
        all_prob.append(probabilities)
        all_y.append(yb.numpy())
    return np.concatenate(all_y), np.concatenate(all_prob)

trained_models, results, predictions = {}, [], {}
for name in ['MLP', 'CNN']:
    payload = torch.load(RUN/'checkpoints'/f'{name}_best.pt', map_location=DEVICE, weights_only=True)
    model = MODEL_CLASSES[name]().to(DEVICE)
    model.load_state_dict(payload['state_dict'])
    trained_models[name] = model.eval()
    y, prob = collect_predictions(model)
    pred = prob.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(10))
    accuracy = accuracy_score(y, pred)
    assert len(y) == 10000 and cm.sum() == len(y)
    assert np.isclose(np.trace(cm)/len(y), accuracy)  # Hình và bảng phải khớp.
    assert np.array_equal(cm.sum(axis=1), np.bincount(y, minlength=10))
    report = classification_report(y, pred, labels=np.arange(10), target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    save_json(RUN/'metrics'/f'{name}_classification_report.json', report)
    rows = pd.DataFrame({'sample_id': np.arange(len(y)), 'true_label': y, 'predicted_label': pred,
                         'run_id': RUN_ID, 'checkpoint_hash': checkpoint_hashes[name]})
    for k in range(10):
        rows[f'prob_{k}'] = prob[:, k]
    rows.to_csv(RUN/'metrics'/f'{name}_predictions.csv', index=False)
    np.save(RUN/'metrics'/f'{name}_confusion.npy', cm)
    predictions[name] = (y, pred, prob, cm)
    results.append(dict(model=name, accuracy=accuracy, macro_f1=f1_score(y, pred, average='macro'),
                        best_epoch=payload['epoch'], **{k: training_info[name][k] for k in ['parameters', 'train_seconds']}))
summary = pd.DataFrame(results)
summary.to_csv(RUN/'metrics'/'summary.csv', index=False)
display(summary)
print('EVALUATION_COMPLETE: bảng và đường chéo confusion matrix đã khớp.')


# %%
for name, (y, pred, prob, cm) in predictions.items():
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted / Dự đoán', ylabel='True / Nhãn thật', title=f'{name}: accuracy={np.trace(cm)/cm.sum():.4f}')
    plt.setp(ax.get_xticklabels(), rotation=65, ha='right')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=8,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout()
    fig.savefig(RUN/'figures'/f'{name}_confusion.png', dpi=140)
    plt.show()

y, pred, prob, cm = predictions['CNN']
display(pd.DataFrame(json.loads((RUN/'metrics'/'CNN_classification_report.json').read_text())).T)
off_diagonal = cm.copy()
np.fill_diagonal(off_diagonal, 0)  # Bỏ các dự đoán đúng để tìm nhầm lẫn.
for flat in np.argsort(off_diagonal.ravel())[-5:][::-1]:
    i, j = np.unravel_index(flat, off_diagonal.shape)
    if off_diagonal[i,j] > 0:
        print(f'Thật {CLASSES[i]} -> đoán {CLASSES[j]}: {off_diagonal[i,j]} ảnh')
wrong = np.flatnonzero(y != pred)[:12]  # Quy tắc chọn rõ ràng: 12 lỗi đầu theo ID.
fig, axes = plt.subplots(3, 4, figsize=(13, 9))
for ax in axes.flat:
    ax.axis('off')
for ax, idx in zip(axes.flat, wrong):
    img, _ = test_ds[int(idx)]
    ax.imshow((img[0]*STD+MEAN).clamp(0,1), cmap='gray')
    ax.set_title(f'ID {idx}: {CLASSES[y[idx]]}\nĐoán {CLASSES[pred[idx]]}: {prob[idx,pred[idx]]:.1%}', fontsize=9)
plt.tight_layout()
fig.savefig(RUN/'figures'/'CNN_errors.png', dpi=140)
plt.show()


# %%
sample_id = 0  # Thay ID từ 0 đến 9999 để tự khám phá.
img, label = test_ds[sample_id]
cnn = trained_models['CNN']
with torch.no_grad():
    first = cnn.features[:2](img.unsqueeze(0).to(DEVICE))  # Thêm batch dimension.
    second = cnn.features[2:5](first)  # Pool rồi Conv thứ hai và ReLU.
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for ax in axes.flat:
    ax.axis('off')
axes[0,0].imshow((img[0]*STD+MEAN).clamp(0,1), cmap='gray')
axes[0,0].set_title(f'ID {sample_id}: {CLASSES[label]}', fontsize=8)
for row, maps in [(1,first), (2,second)]:
    for k in range(8):
        axes[row,k].imshow(maps[0,k].cpu(), cmap='viridis')
        axes[row,k].set_title(f'Conv {row}, kênh {k}', fontsize=8)
plt.tight_layout()
fig.savefig(RUN/'figures'/f'CNN_features_id{sample_id}.png', dpi=140)
plt.show()
save_json(RUN/'metrics'/'feature_view.json', dict(sample_id=sample_id,
    layers=['features.1', 'features.4'], checkpoint_hash=checkpoint_hashes['CNN'], scale='per channel'))


# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow((img[0]*STD+MEAN).clamp(0,1), cmap='gray')
axes[0].set_title(f'ID {sample_id} — thật: {CLASSES[label]}')
axes[0].axis('off')
for ax, name in zip(axes[1:], ['MLP', 'CNN']):
    scores = predictions[name][2][sample_id]
    ax.barh(CLASSES, scores)
    ax.set(xlim=(0,1), title=f'{name}: {CLASSES[scores.argmax()]}', xlabel='Điểm Softmax')
plt.tight_layout()
fig.savefig(RUN/'figures'/f'demo_id{sample_id}.png', dpi=140)
plt.show()
save_json(RUN/'STATUS.json', dict(stage='completed_basic_learning', run_id=RUN_ID,
    trained=['MLP','CNN'], epochs=EPOCHS, test_samples=10000,
    limits=['one seed', 'five epochs', 'no ablation', 'no robustness benchmark'],
    consistency_checks='PASS: matrix total, diagonal accuracy, row support'))
archive = shutil.make_archive(str(ROOT/RUN_ID), 'zip', root_dir=RUN)
print('Gói kết quả:', archive)
print('RUN_COMPLETE', summary.to_dict(orient='records'))


# %%
from google.colab import files  # Tiện ích tải tệp của Colab.
files.download(archive)  # Tải checkpoint, predictions, metrics, figures và cấu hình.
