"""MLP/CNN gốc và các biến thể để so sánh có kiểm soát."""

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
            nn.Linear(32 * 7 * 7, 128),  # Tổng hợp đặc trưng.
            nn.ReLU(),
            nn.Linear(128, 10),  # Điểm số 10 lớp.
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNPlus(nn.Module):
    """Hai khối tích chập, kết hợp BatchNorm và Dropout."""

    def __init__(self):
        super().__init__()
        layers = []
        for incoming, outgoing in [(1, 32), (32, 64)]:
            layers.extend(
                [
                    nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                    nn.BatchNorm2d(outgoing),
                    nn.ReLU(),
                    nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                    nn.BatchNorm2d(outgoing),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNDropout(CNN):
    """Giữ CNN gốc, chỉ thêm Dropout để đo riêng tác động."""

    def __init__(self):
        super().__init__()
        layers = list(self.classifier.children())
        self.classifier = nn.Sequential(*layers[:-1], nn.Dropout(0.3), layers[-1])


class CNNBatchNorm(CNN):
    """Giữ CNN gốc, chỉ thêm BatchNorm sau hai convolution."""

    def __init__(self):
        super().__init__()
        layers = []
        for layer in self.features:
            layers.append(layer)
            if isinstance(layer, nn.Conv2d):
                layers.append(nn.BatchNorm2d(layer.out_channels))
        self.features = nn.Sequential(*layers)


MODEL_CLASSES = {
    "MLP": MLP,
    "CNN": CNN,
    "CNNPlus": CNNPlus,
    "CNNDropout": CNNDropout,
    "CNNBatchNorm": CNNBatchNorm,
}
