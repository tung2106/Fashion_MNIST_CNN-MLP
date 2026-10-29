from pathlib import Path
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Tìm thư mục dữ liệu chung từ vị trí tệp, dù chạy từ thư mục nào.
DATA_ROOT = Path(__file__).resolve().parents[1] / "DU_LIEU"

# Tải dữ liệu và chuyển ảnh thành tensor — một cấu trúc chứa số.
dataset = datasets.FashionMNIST(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Lấy một ảnh và đáp án tương ứng.
image, label = dataset[1]

print("Số ảnh:", len(dataset))
print("Kích thước ảnh:", image.shape)
print("Mã nhãn:", label)
print("Tên loại trang phục:", dataset.classes[label])

# Xem một vùng 5 × 5 pixel ở giữa ảnh.
print("Giá trị pixel ở giữa ảnh:")
print(image[0, 10:15, 10:15])

# Đếm số ảnh mỗi loại và in ra terminal trước khi mở cửa sổ ảnh.
print("\nSố ảnh của từng loại trong tập train gốc:")
labels = dataset.targets

for class_id, class_name in enumerate(dataset.classes):
    count = (labels == class_id).sum().item()
    print(f"{class_id}: {class_name} — {count} ảnh")

# Hiển thị ảnh. plt.show() có thể chờ đến khi bạn đóng cửa sổ ảnh.
plt.imshow(image[0], cmap="gray")
plt.title(dataset.classes[label])
plt.axis("off")
plt.show()

