"""Giữ nguyên kiến trúc MLP/CNN từ notebook giai đoạn 2."""
from torch import nn

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
        return self.layers(x)

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
