"""Tạo notebook mới; không mở, sửa hoặc chạy notebook cũ."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []

def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.strip()+'\n'))

def lesson(title, explanation, code, observe):
    md('## '+title+'\n\n'+explanation)
    compile(code, title, 'exec')
    cells.append(dict(cell_type='code', metadata={}, source=code.strip()+'\n', execution_count=None, outputs=[]))
    md('**Quan sát / tự kiểm tra:** '+observe)

md('''# Fashion-MNIST — Học chậm Python, MLP và CNN

Notebook học tập mới, độc lập với [bản đã chạy](https://colab.research.google.com/drive/1himC9AFXIiUB5KNYW5PlEPnTs7OaOtvL).
Không chép kết quả cũ thành kết quả mới. Không sửa notebook cũ hoặc thư mục kết quả cũ.

**Cách học:** đọc mục đích → đoán đầu ra → chạy một ô bằng Shift+Enter → đọc kết quả và câu hỏi.
Mỗi dòng thao tác có chú thích sau `#`. Các dòng đóng ngoặc chỉ kết thúc biểu thức nhiều dòng.
Chạy từ trên xuống trong cùng phiên. `NameError` thường do chưa chạy ô định nghĩa biến trước đó.
`def` và `class` tạo hàm/lớp, chưa thực hiện huấn luyện; phải gọi chúng mới chạy.

**Lộ trình:** A. Python và dữ liệu (1–12) → B. Mô hình (13–16) → C. Một bước học (17–21) → D. Train/validation (22–27) → E. Test và phân tích (28–32) → F. Lưu (33).
Dừng sau phần A trong buổi đầu. Không cần Run all để học. Từ bước 26 mới huấn luyện đầy đủ.

**Phạm vi giữ nguyên:** 54k/6k/10k; seed 42; batch 128; MLP 784→128→10; CNN 1→16→32 rồi Linear 128→10; Adam 0.001, 5 epoch; chọn validation loss thấp nhất. Không có augmentation/Dropout/BatchNorm. Cùng cấu hình không đảm bảo kết quả giống từng bit trên mọi môi trường.

**Colab:** CPU đủ học dữ liệu; GPU T4 nếu có giúp train nhanh hơn. Runtime là máy tạm; notebook trên Drive và file trong `/content` là hai nơi khác nhau. Bản này xuất kết quả ở `/content/fashion_mnist_hoc_cham`, tách khỏi bản cũ.
''')
lesson('1. Biến, danh sách và chỉ số trong Python',
'''`=` gán giá trị; `==` so sánh. Chuỗi đặt trong dấu nháy. List dùng `[]`, chỉ số bắt đầu từ 0. `print` hiển thị; `len` đếm phần tử. `#` mở chú thích, máy không thực hiện phần sau dấu này.''',
'''ten_lop_vi_du = ['áo', 'quần', 'túi']  # List minh họa, không thay bảng 10 nhãn thật.
vi_tri = 2  # Biến số nguyên chỉ vị trí thứ ba.
print(ten_lop_vi_du[vi_tri])  # Lấy phần tử thứ ba: túi.
print(len(ten_lop_vi_du))  # Đếm ba phần tử.
print(ten_lop_vi_du[:2])  # Cắt từ đầu đến trước vị trí 2: áo, quần.''',
'Kết quả: túi, 3, rồi danh sách áo/quần. `[:2]` không bao gồm phần tử số 2.')
lesson('2. Nạp công cụ làm việc với ảnh',
'''Thư viện là bộ mã viết sẵn. `import X` nạp X; `as` đặt tên ngắn; `from X import Y` lấy Y từ X. NumPy xử lý mảng; tensor là cấu trúc chứa số nhiều chiều của PyTorch; Matplotlib vẽ hình.''',
'''import numpy as np  # Dùng tên np cho NumPy.
import torch  # Công cụ tensor và đạo hàm tự động.
from torch import nn  # Các lớp mạng và hàm loss.
from torchvision import datasets, transforms  # Bộ dữ liệu và phép xử lý ảnh.
from torch.utils.data import DataLoader, Subset  # Gom batch và chọn tập con.
from sklearn.model_selection import train_test_split  # Chia danh sách theo nhãn.
import matplotlib.pyplot as plt  # Vẽ ảnh/biểu đồ.
from pathlib import Path  # Biểu diễn đường dẫn thư mục.
print('PyTorch:', torch.__version__)  # Thuộc tính __version__ là phiên bản thư viện.''',
'Nếu import lỗi, dừng ở đây để sửa môi trường; không chạy ô phụ thuộc. Colab thường có sẵn các thư viện này.')
lesson('3. Cấu hình và thư mục riêng',
'''Viết hoa tên biến là quy ước cho cấu hình, không làm biến bất biến. Batch là nhóm ảnh; epoch là một lượt qua train; learning rate điều khiển quy mô cập nhật. `Path / tên` ghép đường dẫn. UUID tạo tên lần chạy riêng. `mkdir` tạo thư mục.''',
'''from uuid import uuid4  # Sinh mã để tách riêng mỗi lần chạy.
SEED = 42  # Giữ cách chia/ngẫu nhiên dễ lặp lại; không phải trọng số 42.
BATCH_SIZE = 128  # Số ảnh trong một batch đầy.
EPOCHS = 5  # Số lượt qua toàn bộ train khi đến phần huấn luyện.
LR = 0.001  # Learning rate của Adam.
ROOT = Path('/content/fashion_mnist_hoc_cham')  # Gốc mới, không dùng đường dẫn cũ.
DATA = ROOT / 'data'  # Nơi tải ảnh/nhãn.
RUN = ROOT / 'runs' / uuid4().hex  # Tên thư mục kết quả riêng.
RUN.mkdir(parents=True, exist_ok=False)  # Tạo cả thư mục cha; không ghi đè run cũ.
print('Kết quả sẽ lưu tại:', RUN)  # Ghi nhớ đường dẫn này.''',
'Chạy lại ô này tạo run mới: phải chạy lại các bước phụ thuộc để kết quả thuộc cùng run. Các phần tạo file ở cuối nhằm giữ việc học ban đầu đơn giản.')
lesson('4. Chọn thiết bị và đặt seed',
'''CPU là bộ xử lý thông thường, GPU xử lý nhiều phép tính song song; CUDA là nền tảng dùng GPU NVIDIA. `if/else` chọn nhánh. Thụt đầu dòng xác định khối lệnh. `def` định nghĩa hàm; code bên trong chỉ chạy khi gọi hàm.''',
'''import random  # Bộ sinh ngẫu nhiên của Python.
if torch.cuda.is_available():  # Kiểm tra có CUDA sử dụng được không.
    DEVICE = torch.device('cuda')  # Chọn GPU.
else:  # Trường hợp không có CUDA.
    DEVICE = torch.device('cpu')  # Chọn CPU.
def seed_all(seed):  # Hàm nhận một số seed.
    random.seed(seed)  # Đặt nguồn ngẫu nhiên Python.
    np.random.seed(seed)  # Đặt nguồn ngẫu nhiên NumPy.
    torch.manual_seed(seed)  # Đặt nguồn ngẫu nhiên PyTorch.
    torch.backends.cudnn.benchmark = False  # Không tự chọn thuật toán qua đo tốc độ.
    torch.backends.cudnn.deterministic = True  # Ưu tiên cách tính xác định khi hỗ trợ.
seed_all(SEED)  # Gọi hàm; có thể gọi lại trước mỗi mô hình.
print('Thiết bị:', DEVICE)  # In cpu hoặc cuda.''',
'Seed giúp lặp lại trong cùng điều kiện, không bảo đảm giống bit trên mọi thiết bị/phiên bản.')
lesson('5. Tải train gốc và đọc một mẫu',
'''Dataset là bộ dữ liệu. `train=True` chọn phần 60k ảnh, KHÔNG bắt đầu train. `download=True` cho phép tải khi chưa có. Mẫu trả về hai phần; Python nhận chúng vào hai biến bằng phép unpacking.''',
'''raw_train = datasets.FashionMNIST(root=DATA, train=True, download=True)  # Lấy 60k mẫu.
CLASSES = raw_train.classes  # Danh sách tên 10 lớp theo thứ tự nhãn.
image, label = raw_train[0]  # Ảnh đầu tiên và nhãn thật của nó.
print('Số mẫu:', len(raw_train))  # Kỳ vọng 60000.
print('Nhãn:', label, 'Tên:', CLASSES[label])  # Dùng nhãn để tìm tên lớp.
plt.imshow(image, cmap='gray')  # Vẽ ảnh theo thang xám.
plt.axis('off')  # Ẩn trục tọa độ để nhìn ảnh rõ hơn.
plt.show()  # Hiển thị hình.''',
'Chỉ số mẫu 0 không có nghĩa nhãn 0. Ảnh xám có một kênh và 28×28 pixel.')
lesson('6. Pixel, tensor, shape và dtype',
'''Pixel là dữ liệu đầu vào, không phải trọng số. `shape` là kích thước; `dtype` là kiểu số. `ToTensor` chuyển ảnh PIL uint8 thành tensor float32 [1,28,28] ở thang 0–1. `.item()` lấy số Python từ tensor một phần tử.''',
'''to_tensor = transforms.ToTensor()  # Tạo phép chuyển ảnh thành tensor.
pixel_tensor = to_tensor(image)  # Áp dụng phép chuyển vào ảnh vừa lấy.
print(pixel_tensor.shape)  # 1 kênh, cao 28, rộng 28.
print(pixel_tensor.dtype)  # Số thực float32.
print(pixel_tensor.min().item(), pixel_tensor.max().item())  # Khoảng giá trị của ảnh.
print(pixel_tensor[0, 10:15, 10:15])  # Kênh 0, hàng 10–14, cột 10–14.''',
'Dấu chấm không có () như `.shape` đọc thuộc tính; có () như `.min()` gọi phương thức. Đây chưa phải normalize.')
lesson('7. Chia 54k train và 6k validation',
'''Index là vị trí ảnh, label là loại ảnh. Chỉ chia danh sách vị trí, không cắt ảnh. Đối số có tên `test_size` ở hàm này được dùng để tạo validation; test 10k là bộ gốc riêng. `stratify` giữ tỷ lệ lớp; `random_state` cố định cách chia.''',
'''indices = np.arange(len(raw_train))  # Tạo dãy 0, 1, ..., 59999.
all_labels = raw_train.targets.numpy()  # Chuyển tensor nhãn thành mảng NumPy.
train_idx, val_idx = train_test_split(  # Nhận hai mảng vị trí trả về.
    indices,  # Danh sách cần chia.
    test_size=6000,  # Dành 6000 vị trí cho validation.
    random_state=SEED,  # Cùng seed và điều kiện thì lặp lại cách chia.
    stratify=all_labels  # Giữ tỷ lệ mỗi lớp.
)
print(len(train_idx), len(val_idx))  # Kỳ vọng 54000 và 6000.''',
'Hãy giải thích vì sao hàm có chữ test nhưng ở đây tạo validation. Không dùng tập 10k để lựa chọn mô hình.')
lesson('8. Kiểm tra cách chia',
'''`assert` dừng chương trình nếu điều kiện sai. `set` tạo tập hợp không lặp; `intersection` tìm phần chung; `union` gộp tập hợp. Kiểm tra chỉ số không chồng lấn không đồng nghĩa kiểm tra mọi ảnh giống nội dung.''',
'''train_set = set(train_idx.tolist())  # Chuyển mảng thành list rồi tập hợp chỉ số.
val_set = set(val_idx.tolist())  # Tập hợp chỉ số validation.
assert len(train_set) == 54000  # Đủ train và không có chỉ số lặp.
assert len(val_set) == 6000  # Đủ validation và không lặp.
assert len(train_set.intersection(val_set)) == 0  # Hai tập không chung chỉ số.
assert train_set.union(val_set) == set(indices.tolist())  # Không bỏ sót ảnh gốc.
for class_id in range(10):  # Lặp nhãn từ 0 đến 9.
    train_count = (all_labels[train_idx] == class_id).sum()  # Đếm nhãn trong train.
    val_count = (all_labels[val_idx] == class_id).sum()  # Đếm nhãn trong validation.
    print(CLASSES[class_id], train_count, val_count)  # Mỗi lớp 5400 / 600.''',
'`==` trên mảng cho một mảng True/False; sum đếm số True. Kỳ vọng mỗi lớp 5400 train, 600 validation.')
lesson('9. Mean và standard deviation',
'''Mean là độ sáng trung bình. Standard deviation (std) đo mức phân tán quanh trung bình. Chỉ tính từ 54k train, tránh để dữ liệu kiểm tra tham gia thiết kế tiền xử lý. `float()` đổi sang số thực; `/255` đưa pixel gốc về 0–1.''',
'''train_pixels = raw_train.data[train_idx].float() / 255.0  # Chỉ lấy pixel của train.
MEAN = train_pixels.mean().item()  # Trung bình của toàn bộ pixel train.
STD = train_pixels.std(unbiased=False).item()  # Độ lệch chuẩn tổng thể.
assert STD > 0  # Không được chia cho 0 khi chuẩn hóa.
del train_pixels  # Bỏ mảng tạm để giảm dùng bộ nhớ.
print('Mean:', MEAN, 'Std:', STD)  # Thường xấp xỉ 0.2862 và 0.3532 với split này.''',
'Mean/std không phải tham số trọng số được optimizer học. Chúng là thống kê cố định từ dữ liệu train.')
lesson('10. Normalize ảnh và tạo các tập con',
'''Normalize tính `(pixel - mean) / std`; kết quả có thể âm hoặc lớn hơn 1. Compose ghép các phép biến đổi theo thứ tự. `(MEAN,)` là tuple một phần tử cho một kênh xám. Subset chọn mẫu theo chỉ số, không tạo nhãn mới.''',
'''transform = transforms.Compose([  # Các phép biến đổi được chạy lần lượt.
    transforms.ToTensor(),  # Ảnh → tensor ở thang 0–1.
    transforms.Normalize((MEAN,), (STD,))  # Đổi tâm và thang đo.
])
source = datasets.FashionMNIST(root=DATA, train=True, transform=transform)  # Cùng train gốc.
train_ds = Subset(source, train_idx.tolist())  # 54k mẫu dành học trọng số.
val_ds = Subset(source, val_idx.tolist())  # 6k mẫu dành chọn phiên bản mô hình.
test_ds = datasets.FashionMNIST(root=DATA, train=False, download=True, transform=transform)  # 10k riêng.
assert len(test_ds) == 10000  # Chỉ kiểm tra số mẫu, chưa đánh giá mô hình.
normalized_image, normalized_label = train_ds[0]  # Một mẫu trong train đã chuẩn hóa.
print(normalized_image.shape, normalized_label)  # Shape vẫn là [1,28,28].''',
'Pixel thay đổi vì tiền xử lý; optimizer không sửa pixel. Tất cả tập dùng cùng mean/std từ train.')
lesson('11. Tạo DataLoader',
'''Dataset cung cấp từng mẫu; DataLoader gom mẫu thành batch. `shuffle=True` xáo thứ tự khi duyệt; không đổi tập thành viên của split. `num_workers=0` đọc dữ liệu trong tiến trình hiện tại. Hàm tạo loader mới giúp hai mô hình bắt đầu cùng thứ tự ảnh.''',
'''def make_train_loader():  # Chưa nạp batch cho đến khi gọi và duyệt loader.
    generator = torch.Generator()  # Bộ sinh số ngẫu nhiên riêng.
    generator.manual_seed(SEED)  # Đặt seed cho thứ tự batch.
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)  # Gom ảnh.
    return loader  # Trả loader cho nơi gọi hàm.
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)  # Không cần đảo validation.
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)  # Giữ thứ tự để lưu ID test.''',
'Tạo loader không bắt đầu huấn luyện. Phải có vòng lặp và optimizer mới học trọng số.')
lesson('12. Lấy một batch — điểm dừng buổi đầu',
'''`iter` tạo bộ duyệt; `next` lấy phần tiếp theo. B là số ảnh trong batch; [B,1,28,28] là ảnh; [B] là nhãn. `.clamp(0,1)` giới hạn giá trị hiển thị. Hoàn tác normalize bằng nhân std rồi cộng mean.''',
'''train_loader = make_train_loader()  # Tạo loader train.
batch_iterator = iter(train_loader)  # Tạo bộ duyệt các batch.
images, labels = next(batch_iterator)  # Lấy batch đầu vào hai biến.
print('Ảnh:', images.shape, 'Nhãn:', labels.shape)  # [128,1,28,28] và [128].
print('Số batch:', len(train_loader))  # 422 nếu giữ batch cuối 112 ảnh.
view_image = images[0, 0] * STD + MEAN  # Ảnh đầu, kênh đầu, về thang 0–1.
plt.imshow(view_image.clamp(0, 1), cmap='gray')  # Hiển thị ảnh đầu.
plt.title(CLASSES[labels[0].item()])  # Lấy nhãn rồi tìm tên lớp.
plt.show()  # Vẽ ra màn hình.''',
'DỪNG ở đây nếu mới học. Tự giải thích 128, 1, 28, 28 và vì sao chỉ có 128 nhãn. Đáp án: mỗi ảnh có một nhãn; nhãn không phải một bảng pixel.')
lesson('13. Trọng số là hệ số nhân',
'''Ví dụ độc lập: y = x × w; x là pixel giả lập, w là trọng số. requires_grad=True yêu cầu theo dõi w để tính đạo hàm. Loss bình phương chỉ dùng cho ví dụ này, không thay CrossEntropy của bài phân loại. Gradient = 2×(xw−1)×x, tại đây bằng −0.96.''',
'''x_demo = torch.tensor(0.8)  # Dữ liệu ví dụ, không cần gradient.
w_demo = torch.tensor(0.5, requires_grad=True)  # Trọng số cần học.
y_demo = x_demo * w_demo  # Dự đoán 0.4.
loss_demo = (y_demo - 1.0) ** 2  # **2 là bình phương sai lệch: 0.36.
loss_demo.backward()  # Tính đạo hàm loss theo w_demo.
print('Dự đoán:', y_demo.item(), 'Loss:', loss_demo.item())  # Xem giá trị.
print('Gradient:', w_demo.grad.item())  # Kỳ vọng xấp xỉ -0.96.
with torch.no_grad():  # Không tạo đồ thị gradient cho phép cập nhật minh họa.
    w_demo -= 0.1 * w_demo.grad  # SGD đơn giản: w = w - learning_rate × gradient.
print('Trọng số mới:', w_demo.item())  # Khoảng 0.596; pixel vẫn 0.8.''',
'Gradient âm làm SGD cơ bản tăng trọng số. Adam còn dùng lịch sử gradient nên không suy luận hướng chỉ từ dấu gradient hiện tại trong mọi bước.')
lesson('14. Định nghĩa MLP bằng class',
'''Class là khuôn tạo đối tượng; `MLP(nn.Module)` kế thừa lớp mạng của PyTorch. `self` là đối tượng hiện tại. `__init__` chạy khi tạo model; `super().__init__()` khởi tạo phần lớp cha. `forward` mô tả đường đi dữ liệu; gọi model(x) sẽ chạy forward. Sequential xếp các lớp theo thứ tự. Trọng số/bias được PyTorch khởi tạo, chưa học.''',
'''class MLP(nn.Module):  # Khai báo một loại mô hình.
    def __init__(self):  # Hàm dựng khi gọi MLP().
        super().__init__()  # Bật cơ chế quản lý tham số của nn.Module.
        self.layers = nn.Sequential(  # Các lớp sẽ chạy nối tiếp.
            nn.Flatten(),  # [B,1,28,28] → [B,784]; giữ thứ tự giá trị.
            nn.Linear(784, 128),  # Mỗi ảnh tạo 128 giá trị từ 784 đầu vào, có bias.
            nn.ReLU(),  # Số âm → 0; số dương giữ nguyên; tạo tính phi tuyến.
            nn.Linear(128, 10)  # Tạo 10 logits tương ứng 10 lớp.
        )
    def forward(self, x):  # x là tensor đầu vào.
        return self.layers(x)  # Trả kết quả đi qua toàn bộ lớp.
mlp_probe = MLP()  # Tạo đối tượng để quan sát, chưa train.
print(mlp_probe(images[:2]).shape)  # Hai ảnh → [2,10].''',
'128 trong Linear là số nút/đặc trưng, không phải batch size. MLP nhận batch kích thước khác vẫn được.')
lesson('15. Định nghĩa CNN',
'''Conv2d: tích chập trên ảnh 2D, dùng chung bộ lọc ở nhiều vị trí. `kernel_size=3`: vùng 3×3. `padding=1`: thêm viền 0; với stride mặc định 1 giữ chiều rộng/cao. MaxPool2d(2) lấy số lớn nhất trong vùng 2×2, giảm mỗi chiều một nửa. Feature map là bảng phản ứng của bộ lọc, không mặc định tương ứng bộ phận áo cụ thể.''',
'''class CNN(nn.Module):  # Khuôn mô hình CNN.
    def __init__(self):  # Khởi tạo lớp và các tham số.
        super().__init__()  # Khởi tạo nn.Module.
        self.features = nn.Sequential(  # Phần học đặc trưng từ ảnh.
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 1 kênh → 16 kênh, 28×28.
            nn.ReLU(),  # Giữ giá trị dương.
            nn.MaxPool2d(2),  # 28×28 → 14×14.
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16 → 32 kênh, 14×14.
            nn.ReLU(),  # Phi tuyến sau convolution thứ hai.
            nn.MaxPool2d(2)  # 14×14 → 7×7.
        )
        self.classifier = nn.Sequential(  # Phần chuyển đặc trưng thành điểm lớp.
            nn.Flatten(),  # [B,32,7,7] → [B,1568].
            nn.Linear(1568, 128),  # Tổng hợp thành 128 giá trị.
            nn.ReLU(),  # Thêm tính phi tuyến.
            nn.Linear(128, 10)  # Đầu ra 10 logits, chưa softmax.
        )
    def forward(self, x):  # Đường đi khi gọi model(x).
        features = self.features(x)  # Xử lý ảnh qua bộ lọc và pooling.
        logits = self.classifier(features)  # Chuyển đặc trưng thành điểm lớp.
        return logits  # Trả điểm cho nơi gọi.''',
'CNN cũng Flatten, nhưng ở gần cuối sau khi học đặc trưng. Không đảm bảo CNN luôn ít tham số hơn MLP.')
lesson('16. Theo dõi shape qua từng lớp',
'''`for` duyệt lần lượt; `list` chuyển các lớp thành danh sách; dấu + ghép hai danh sách. `type(...).__name__` lấy tên loại lớp. Shape thay đổi nhưng trục B phải giữ nguyên.''',
'''cnn_probe = CNN()  # Mô hình quan sát riêng, chưa train.
probe_image = images[:1]  # Lấy một ảnh, giữ trục batch [1,1,28,28].
layers = list(cnn_probe.features) + list(cnn_probe.classifier)  # Các lớp theo thứ tự.
with torch.no_grad():  # Chỉ xem đầu ra, không cần lưu đồ thị đạo hàm.
    for layer in layers:  # Lần lượt chạy từng lớp.
        probe_image = layer(probe_image)  # Đầu ra lớp trước là đầu vào lớp sau.
        print(type(layer).__name__, tuple(probe_image.shape))  # In tên và shape.''',
'Sau Pool lần hai: [1,32,7,7]; sau Flatten: [1,1568]; cuối: [1,10].')
lesson('17. Chuẩn bị mô hình cho một bước học',
'''Optimizer sửa tham số dựa trên gradient. Adam dùng cả gradient và các trung bình động. `.to(DEVICE)` chuyển tensor/mô hình tới cùng thiết bị; ảnh và model phải cùng nơi. Model minh họa riêng không dùng làm khởi đầu cho thí nghiệm chính.''',
'''seed_all(SEED)  # Cố định khởi tạo ví dụ.
demo_model = MLP().to(DEVICE)  # Mô hình riêng cho bài học này.
demo_optimizer = torch.optim.Adam(demo_model.parameters(), lr=LR)  # Tham số nào cần tối ưu.
criterion = nn.CrossEntropyLoss()  # Hàm loss cho phân loại một nhãn trong 10 lớp.
xb = images[:16].to(DEVICE)  # Lấy 16 ảnh để phép minh họa nhỏ.
yb = labels[:16].to(DEVICE)  # Nhãn tương ứng; vẫn là số nguyên 0–9.
before = demo_model.layers[1].weight.detach().clone()  # Chụp bản sao trọng số Linear đầu.''',
'detach tách tensor khỏi đồ thị đạo hàm; clone sao chép dữ liệu để cập nhật không làm bản chụp thay đổi.')
lesson('18. Forward và logits',
'''Forward là tính đầu ra từ đầu vào bằng trọng số hiện tại. Logits là điểm lớp, có thể âm hoặc dương, không phải pixel hay phần trăm. Softmax chuyển 10 điểm thành xác suất cộng lại bằng 1; argmax lấy vị trí có điểm cao nhất.''',
'''demo_optimizer.zero_grad()  # Xóa gradient cũ; không xóa trọng số.
logits = demo_model(xb)  # Forward: 16 ảnh → [16,10].
probabilities = logits.softmax(dim=1)  # Chỉ để quan sát; dim=1 là trục lớp.
predicted_labels = logits.argmax(dim=1)  # Một mã nhãn dự đoán cho mỗi ảnh.
print('Logits ảnh đầu:', logits[0].detach().cpu())  # Tách khỏi đồ thị, đưa về CPU để xem.
print('Tổng xác suất:', probabilities[0].sum().item())  # Xấp xỉ 1.
print('Đoán / thật:', predicted_labels[0].item(), yb[0].item())  # Model chưa học thường sai.''',
'Logits không cần nằm trong 0–1. Softmax không thay đổi lớp có điểm cao nhất.')
lesson('19. Loss đo sai lệch',
'''CrossEntropyLoss nhận logits và nhãn nguyên, bên trong đã xử lý log-softmax ổn định. Không truyền probabilities thay logits. Loss trung bình của batch thấp hơn nghĩa là mô hình gán điểm phù hợp hơn với nhãn, không phải tỷ lệ lỗi.''',
'''loss = criterion(logits, yb)  # Đối chiếu 10 điểm mỗi ảnh với nhãn thật tương ứng.
print('Loss:', loss.item())  # Lấy số để xem; loss=0.3 không có nghĩa sai 30%.''',
'Tại đây chưa thay đổi trọng số. Loss một số, logits [16,10], nhãn [16].')
lesson('20. Backward tính gradient',
'''Gradient là các đạo hàm của loss theo tham số. Nó mô tả độ nhạy cục bộ, không tự sửa trọng số. PyTorch tích lũy gradient nên cần zero_grad trước bước học mới. Không chạy backward lần hai trên cùng đồ thị đã giải phóng; chạy lại từ bước 17 nếu muốn lặp ví dụ.''',
'''loss.backward()  # Tính gradient cho các tham số đã tham gia forward.
gradient = demo_model.layers[1].weight.grad  # Lấy gradient của ma trận trọng số đầu.
print('Gradient của 5 trọng số:', gradient[0, :5].detach().cpu())  # Có thể âm/dương/0.
print('Trọng số còn nguyên:', torch.equal(before, demo_model.layers[1].weight.detach()))  # True.''',
'Dấu gradient chỉ hướng tăng loss cho SGD cơ bản. Không kết luận dự đoán đã tốt hơn chỉ vì backward chạy xong.')
lesson('21. Step cập nhật trọng số',
'''Optimizer mới thực sự sửa trọng số. abs lấy trị tuyệt đối; mean lấy trung bình để đo mức thay đổi. Một lần cập nhật không bảo đảm loss giảm trên mọi batch hoặc mọi ảnh.''',
'''demo_optimizer.step()  # Adam cập nhật các tham số bằng gradient đã tính.
after = demo_model.layers[1].weight.detach()  # Trọng số sau cập nhật.
change = (after - before).abs().mean().item()  # Mức thay đổi tuyệt đối trung bình.
with torch.no_grad():  # Đo lại không tạo gradient mới.
    new_loss = criterion(demo_model(xb), yb).item()  # Forward lại bằng trọng số mới.
print('Thay đổi trọng số:', change)  # Kỳ vọng lớn hơn 0.
print('Loss trước / sau:', loss.item(), new_loss)  # Quan sát kết quả thực.
assert change > 0  # Xác nhận đã có cập nhật trong ví dụ này.''',
'Có thể dừng buổi thứ hai ở đây. Khi bắt đầu thí nghiệm chính sẽ tạo MLP/CNN mới, không tái sử dụng demo_model.')
lesson('22. Hàm train một epoch',
'''Hàm dưới dài hơn các ô trước vì một vòng học phải giữ trong cùng khối thụt dòng. `model.train()` chọn chế độ; không tự cập nhật. `for images, labels in loader` lần lượt nhận từng batch. `+=` cộng dồn. Loss mặc định trung bình batch nên nhân số ảnh trước khi gộp cả epoch. Training accuracy tính khi trọng số liên tục đổi.''',
'''def train_one_epoch(model, loader, optimizer):  # Nhận model, dữ liệu và bộ tối ưu.
    model.train()  # Bật chế độ huấn luyện.
    total_loss = 0.0  # Tổng loss theo ảnh.
    total_correct = 0  # Tổng số dự đoán đúng.
    total_images = 0  # Tổng ảnh đã đi qua.
    for batch_images, batch_labels in loader:  # Lấy một batch mỗi vòng.
        batch_images = batch_images.to(DEVICE)  # Chuyển ảnh tới thiết bị của model.
        batch_labels = batch_labels.to(DEVICE)  # Chuyển nhãn cùng thiết bị.
        optimizer.zero_grad()  # Xóa gradient bước trước.
        batch_logits = model(batch_images)  # Forward.
        batch_loss = criterion(batch_logits, batch_labels)  # Tính loss.
        batch_loss.backward()  # Tính gradient.
        optimizer.step()  # Cập nhật trọng số.
        batch_count = batch_labels.size(0)  # Số ảnh thật, batch cuối có thể ít hơn 128.
        predicted = batch_logits.argmax(dim=1)  # Chọn lớp có điểm cao nhất.
        total_correct += (predicted == batch_labels).sum().item()  # Đếm đúng.
        total_loss += batch_loss.item() * batch_count  # Đổi trung bình batch thành tổng.
        total_images += batch_count  # Đếm tất cả ảnh đã xử lý.
    return total_loss / total_images, total_correct / total_images  # Loss và accuracy epoch.''',
'Ô định nghĩa hàm chưa train. Hãy chỉ đúng hai dòng backward/step.')
lesson('23. Hàm validation không cập nhật',
'''`with` giới hạn một chế độ trong khối code. no_grad tắt ghi đồ thị đạo hàm, tiết kiệm bộ nhớ. `model.eval()` chọn chế độ đánh giá; riêng eval không tự tắt gradient. Validation không gọi optimizer. Đây là đo trên một phiên bản model cố định.''',
'''def evaluate(model, loader):  # Hàm dùng được cho validation và test sau khi chốt.
    model.eval()  # Chế độ đánh giá.
    total_loss = 0.0  # Tổng loss theo ảnh.
    total_correct = 0  # Tổng số đúng.
    total_images = 0  # Tổng số ảnh.
    with torch.no_grad():  # Không lưu đồ thị đạo hàm.
        for batch_images, batch_labels in loader:  # Duyệt từng batch.
            batch_images = batch_images.to(DEVICE)  # Chuyển ảnh.
            batch_labels = batch_labels.to(DEVICE)  # Chuyển nhãn.
            batch_logits = model(batch_images)  # Forward, không học.
            batch_loss = criterion(batch_logits, batch_labels)  # Tính loss để đo.
            batch_count = batch_labels.size(0)  # Kích thước batch thật.
            predicted = batch_logits.argmax(dim=1)  # Nhãn dự đoán.
            total_correct += (predicted == batch_labels).sum().item()  # Đếm đúng.
            total_loss += batch_loss.item() * batch_count  # Cộng tổng loss.
            total_images += batch_count  # Cộng số mẫu.
    return total_loss / total_images, total_correct / total_images  # Hai số trả về.''',
'So sánh với bước 22: không có zero_grad/backward/step. Đây là khác biệt học và đánh giá.')
lesson('24. Chuẩn bị lưu kết quả — phần kỹ thuật phụ trợ',
'''Dictionary dùng `{khóa: giá_trị}`. JSON lưu dữ liệu có cấu trúc dạng chữ; CSV lưu bảng; checkpoint lưu trọng số. state_dict là ánh xạ tên tham số tới tensor. Đoạn này phục vụ truy lại kết quả, không phải thuật toán học.''',
'''import json  # Đọc/ghi JSON.
import pandas as pd  # Tạo bảng và lưu CSV.
import time  # Đo thời gian chạy.
import hashlib  # Tạo dấu vân tay SHA256 của file.
def save_json(path, value):  # Hàm phụ trợ nhận đường dẫn và nội dung.
    text = json.dumps(value, ensure_ascii=False, indent=2)  # Giữ tiếng Việt và căn lề.
    path.write_text(text, encoding='utf-8')  # Ghi chữ ra file.
MODEL_CLASSES = {'MLP': MLP, 'CNN': CNN}  # Tra tên để tạo đúng lớp model.
CONFIG = {'seed': SEED, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'lr': LR, 'device': str(DEVICE), 'torch': str(torch.__version__)}  # Cấu hình thực.
save_json(RUN / 'config.json', CONFIG)  # Lưu cấu hình lần này.
np.savez(RUN / 'splits.npz', train=train_idx, validation=val_idx)  # Lưu chỉ số chia tập.
save_json(RUN / 'preprocessing.json', {'mean': MEAN, 'std': STD, 'classes': CLASSES})  # Lưu chuẩn hóa.''',
'Kết quả mới nằm trong RUN riêng. Không có số liệu test hay trọng số được bịa/sao chép.')
lesson('25. Lặp các epoch và chọn checkpoint',
'''`float("inf")` là vô cực, để loss đầu tiên luôn nhỏ hơn mốc ban đầu. `range(1,EPOCHS+1)` chạy 1 đến EPOCHS. f-string chèn biến vào chuỗi; `:.4f` in 4 chữ số thập phân. Chọn checkpoint theo validation loss, hòa thì giữ epoch sớm hơn. Hàm được giữ trong một ô để không phá khối Python; đọc lần lượt: khởi tạo → học → đo → ghi → chọn.''',
'''def fit(name):  # Huấn luyện một model từ đầu.
    seed_all(SEED)  # Đặt lại seed trước khởi tạo.
    model = MODEL_CLASSES[name]().to(DEVICE)  # Tạo MLP hoặc CNN mới.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Adam riêng cho model này.
    loader = make_train_loader()  # Loader mới, cùng quy tắc thứ tự ảnh.
    best_loss = float('inf')  # Chưa có checkpoint tốt nhất.
    history = []  # List các hàng kết quả epoch.
    start = time.perf_counter()  # Bắt đầu đo thời gian train + validation.
    for epoch in range(1, EPOCHS + 1):  # Lặp đủ số epoch cấu hình.
        train_loss, train_acc = train_one_epoch(model, loader, optimizer)  # Học.
        val_loss, val_acc = evaluate(model, val_loader)  # Đo trên validation.
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'train_accuracy': train_acc, 'val_accuracy': val_acc}  # Một hàng kết quả.
        history.append(row)  # Thêm hàng vào cuối list.
        if val_loss < best_loss:  # Chỉ thay checkpoint khi validation loss tốt hơn.
            best_loss = val_loss  # Cập nhật mốc so sánh.
            payload = {'state_dict': model.state_dict(), 'model_name': name, 'epoch': epoch, 'mean': MEAN, 'std': STD, 'classes': CLASSES, 'config': CONFIG}  # Dữ liệu checkpoint.
            torch.save(payload, RUN / f'{name}_best.pt')  # Lưu ngay, tránh trọng số đổi sau đó.
        pd.DataFrame(history).to_csv(RUN / f'{name}_history.csv', index=False)  # Lưu lịch sử từng epoch.
        print(f'{name} epoch {epoch}: train={train_loss:.4f}, validation={val_loss:.4f}', flush=True)  # Hiện tiến độ.
    elapsed = time.perf_counter() - start  # Số giây đã dùng.
    parameter_count = sum(p.numel() for p in model.parameters())  # Cộng số phần tử tham số.
    info = {'seconds': elapsed, 'parameters': parameter_count, 'best_val_loss': best_loss}  # Tóm tắt lần train.
    save_json(RUN / f'{name}_training.json', info)  # Ghi thông tin thật.
    return info  # Trả cho ô gọi hàm.''',
'sum(p.numel() for p in ...) là cộng số phần tử của từng tensor tham số. Bản học tập chưa hỗ trợ resume chính xác; mất runtime thì chạy lại hoặc dùng checkpoint đã tải.')
lesson('26. Huấn luyện MLP — chỉ chạy khi đã hiểu',
'''Đến đây mới train trên toàn bộ 54k ảnh. Hàm fit tự tạo model mới nên không dùng trọng số của bài minh họa. 5 epoch là bài học cơ bản, không phải cam kết đạt điểm số cụ thể.''',
'''training_info = {}  # Dictionary chứa kết quả từng model.
training_info['MLP'] = fit('MLP')  # Train MLP rồi lưu thông tin theo khóa MLP.''',
'Kỳ vọng in 5 dòng epoch. Nếu lỗi, đọc dòng lỗi đầu tiên và giữ lại thông báo; không coi ô chưa xong là có kết quả.')
lesson('27. Huấn luyện CNN và nhìn đường cong',
'''CNN dùng cùng split, batch size, epoch, optimizer và quy tắc chọn checkpoint. Số tham số vẫn khác nên chưa kiểm soát mọi yếu tố. Overfitting có thể xuất hiện khi train loss giảm mà validation loss tăng qua nhiều epoch.''',
'''training_info['CNN'] = fit('CNN')  # Train CNN mới với cùng quy tắc.
for name in ['MLP', 'CNN']:  # Vẽ từng model.
    history = pd.read_csv(RUN / f'{name}_history.csv')  # Đọc lịch sử thực.
    plt.figure(figsize=(6, 3))  # Tạo hình mới.
    plt.plot(history['epoch'], history['train_loss'], label='train')  # Đường train loss.
    plt.plot(history['epoch'], history['val_loss'], label='validation')  # Đường validation loss.
    plt.xlabel('Epoch')  # Tên trục ngang.
    plt.ylabel('Loss')  # Tên trục dọc.
    plt.title(name)  # Tên model.
    plt.legend()  # Hiện chú giải hai đường.
    plt.tight_layout()  # Căn bố cục.
    plt.savefig(RUN / f'{name}_learning_curve.png')  # Lưu hình từ số liệu này.
    plt.show()  # Hiển thị hình.''',
'Điểm validation tốt hơn train chưa tự chứng minh không overfit: train được đo trong lúc trọng số thay đổi. Hãy tìm epoch có val_loss thấp nhất.')
lesson('28. Chốt model trước khi xem test',
'''SHA256 là dấu vân tay file để nhận diện checkpoint, không phải chỉ số chất lượng. frozen_protocol ghi lựa chọn đã chốt trước test. Test cũ đã được xem trong bản trước; bản này phục vụ học lại, không tuyên bố test hoàn toàn chưa từng được biết.''',
'''checkpoint_hashes = {}  # Dictionary lưu hash mỗi checkpoint.
trained_models = {}  # Chứa model đã nạp trọng số tốt nhất.
for name in ['MLP', 'CNN']:  # Xử lý hai model.
    path = RUN / f'{name}_best.pt'  # Đường dẫn checkpoint theo validation.
    checkpoint_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()  # Hash nội dung file.
    payload = torch.load(path, map_location=DEVICE, weights_only=True)  # Nạp checkpoint do mình tạo.
    model = MODEL_CLASSES[name]().to(DEVICE)  # Tạo kiến trúc tương ứng.
    model.load_state_dict(payload['state_dict'])  # Thay trọng số ngẫu nhiên bằng trọng số đã học.
    trained_models[name] = model.eval()  # Đặt chế độ đánh giá.
save_json(RUN / 'frozen_protocol.json', {'selection': 'minimum validation loss; earliest tie', 'checkpoint_hashes': checkpoint_hashes, 'split_hash': hashlib.sha256((RUN / 'splits.npz').read_bytes()).hexdigest()})  # Ghi bằng chứng chốt.
print('Đã chốt checkpoint; không sửa cấu hình theo test.')  # Mốc trước đánh giá.''',
'Chỉ dùng checkpoint tự tạo/tin cậy. Bước này chưa tính accuracy test.')
lesson('29. Thu thập dự đoán test',
'''append thêm một batch vào list; concatenate ghép các batch theo trục mẫu. `.cpu().numpy()` đưa về CPU rồi chuyển sang NumPy để tính chỉ số. no_grad và eval cùng được dùng. Trật tự test cố định để ID khớp ảnh.''',
'''def predict_test(model):  # Hàm thu nhãn thật và xác suất của toàn bộ test.
    all_labels = []  # List chứa nhãn của các batch.
    all_probabilities = []  # List chứa xác suất của các batch.
    model.eval()  # Chế độ đánh giá.
    with torch.no_grad():  # Không cần đạo hàm.
        for batch_images, batch_labels in test_loader:  # Duyệt 10k ảnh.
            batch_images = batch_images.to(DEVICE)  # Chuyển tới thiết bị của model.
            batch_logits = model(batch_images)  # Tính điểm dự đoán.
            batch_probs = batch_logits.softmax(dim=1)  # Xác suất theo 10 lớp.
            all_probabilities.append(batch_probs.cpu().numpy())  # Lưu batch xác suất.
            all_labels.append(batch_labels.numpy())  # Lưu batch nhãn thật.
    return np.concatenate(all_labels), np.concatenate(all_probabilities)  # Ghép theo mẫu.''',
'Softmax cao không đảm bảo đúng. Không có backward/step khi chạy test.')
lesson('30. So sánh MLP/CNN bằng một nguồn số liệu',
'''Accuracy = số đúng / số mẫu. Confusion matrix: hàng là nhãn thật, cột là dự đoán. Đường chéo là dự đoán đúng. Precision = đúng trong các ảnh đoán là lớp đó; recall = đúng trong các ảnh thật thuộc lớp đó; F1 kết hợp precision/recall; macro-F1 trung bình đều F1 của 10 lớp. CSV predictions là nguồn tính bảng và hình.''',
'''from sklearn.metrics import accuracy_score, f1_score, confusion_matrix  # Hàm tính chỉ số.
predictions = {}  # Lưu dự đoán cho các bước vẽ/phân tích.
rows = []  # Hàng bảng so sánh.
for name in ['MLP', 'CNN']:  # Đánh giá mỗi model đã chốt.
    y_true, probs = predict_test(trained_models[name])  # Lấy nhãn thật và xác suất.
    y_pred = probs.argmax(axis=1)  # Lớp có xác suất cao nhất mỗi ảnh.
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(10))  # Ma trận 10×10.
    accuracy = accuracy_score(y_true, y_pred)  # Tỷ lệ đúng.
    macro_f1 = f1_score(y_true, y_pred, average='macro')  # Trung bình F1 các lớp.
    assert len(y_true) == 10000 and matrix.sum() == 10000  # Kiểm tra đủ mẫu.
    assert np.isclose(np.trace(matrix) / 10000, accuracy)  # Đường chéo khớp accuracy.
    table = pd.DataFrame({'sample_id': np.arange(10000), 'true_label': y_true, 'predicted_label': y_pred, 'checkpoint_hash': checkpoint_hashes[name]})  # Bảng từng ảnh.
    for class_id in range(10):  # Thêm xác suất của từng lớp.
        table[f'prob_{class_id}'] = probs[:, class_id]  # Lấy một cột xác suất.
    table.to_csv(RUN / f'{name}_predictions.csv', index=False)  # Lưu dữ liệu gốc đánh giá.
    predictions[name] = {'true': y_true, 'pred': y_pred, 'probs': probs, 'matrix': matrix}  # Lưu trong bộ nhớ.
    rows.append({'model': name, 'accuracy': accuracy, 'macro_f1': macro_f1, 'parameters': training_info[name]['parameters']})  # Thêm hàng.
summary = pd.DataFrame(rows)  # Tạo bảng so sánh.
summary.to_csv(RUN / 'summary.csv', index=False)  # Ghi bảng kết quả thật.
print(summary.to_string(index=False))  # Hiển thị bảng.
print('EVALUATION_COMPLETE')  # Chỉ in khi tất cả phép kiểm tra trên đã qua.''',
'Các số liệu chỉ có sau khi chạy thật. Không đặt sẵn accuracy của bản cũ trong notebook mới.')
lesson('31. Xem ma trận nhầm lẫn và một ảnh sai',
'''flatnonzero trả vị trí thỏa điều kiện. `!=` là khác nhau. `if len(...) > 0` tránh lỗi nếu không có ảnh sai. Nhãn ảnh và dự đoán phải đọc từ cùng sample_id. Lỗi test để mô tả giới hạn, không dùng chọn cấu hình rồi gọi test độc lập.''',
'''cnn_result = predictions['CNN']  # Kết quả CNN từ bước 30.
plt.figure(figsize=(8, 7))  # Khung hình cho ma trận.
plt.imshow(cnn_result['matrix'], cmap='Blues')  # Màu đậm nghĩa nhiều ảnh.
plt.xticks(range(10), CLASSES, rotation=90)  # Nhãn cột là dự đoán.
plt.yticks(range(10), CLASSES)  # Nhãn hàng là thực tế.
plt.xlabel('Dự đoán')  # Trục ngang.
plt.ylabel('Nhãn thật')  # Trục dọc.
plt.colorbar()  # Thang màu số lượng.
plt.tight_layout()  # Căn hình không cắt tên lớp.
plt.savefig(RUN / 'CNN_confusion.png')  # Lưu hình từ đúng matrix.
plt.show()  # Hiện ma trận.
wrong_ids = np.flatnonzero(cnn_result['true'] != cnn_result['pred'])  # ID bị đoán sai.
if len(wrong_ids) > 0:  # Chỉ lấy ảnh nếu thật sự có lỗi.
    sample_id = int(wrong_ids[0])  # Quy tắc cố định: lỗi đầu tiên theo ID.
    wrong_image, true_label = test_ds[sample_id]  # Lấy ảnh và nhãn tương ứng.
    predicted_label = int(cnn_result['pred'][sample_id])  # Nhãn model đã đoán.
    plt.imshow(wrong_image[0] * STD + MEAN, cmap='gray')  # Hoàn tác chuẩn hóa để xem.
    plt.title(f'Thật: {CLASSES[true_label]} / Đoán: {CLASSES[predicted_label]}')  # Ghi hai nhãn.
    plt.show()  # Hiện ảnh sai.''',
'Hàng Shirt, cột T-shirt/top: áo sơ mi bị đoán thành áo thun. Không đọc ngược hai trục.')
lesson('32. Nhìn đặc trưng CNN và dự đoán cùng ảnh',
'''unsqueeze(0) thêm trục batch: [1,28,28] → [1,1,28,28]. `features[:2]` lấy Conv đầu + ReLU. Đây là phản ứng bộ lọc đã học, không chứng minh một kênh hiểu tay áo. Mỗi kênh có thang màu riêng, không so độ sáng tuyệt đối giữa kênh.''',
'''sample_id = 0  # Chọn ID cố định để học cách xem, không tuning theo ảnh này.
sample_image, sample_label = test_ds[sample_id]  # Lấy cùng ảnh cho MLP và CNN.
one_image_batch = sample_image.unsqueeze(0).to(DEVICE)  # Thêm batch và chuyển thiết bị.
with torch.no_grad():  # Chỉ quan sát đặc trưng.
    feature_maps = trained_models['CNN'].features[:2](one_image_batch)  # Conv đầu + ReLU.
fig, axes = plt.subplots(1, 5, figsize=(12, 3))  # 1 ảnh gốc và 4 bản đồ.
axes[0].imshow(sample_image[0] * STD + MEAN, cmap='gray')  # Ảnh đã hoàn tác chuẩn hóa.
axes[0].set_title(CLASSES[sample_label])  # Nhãn thật.
for channel in range(4):  # Xem bốn kênh đầu.
    axes[channel + 1].imshow(feature_maps[0, channel].cpu(), cmap='viridis')  # Bản đồ của ảnh đầu.
    axes[channel + 1].set_title(f'Kênh {channel}')  # Tên kênh, không gán nghĩa bộ phận.
for ax in axes:  # Làm sạch trục hiển thị.
    ax.axis('off')  # Ẩn trục từng ô.
plt.tight_layout()  # Căn bố cục.
plt.savefig(RUN / 'CNN_feature_maps.png')  # Lưu hình.
plt.show()  # Hiện hình.
for name in ['MLP', 'CNN']:  # Đọc dự đoán của cùng sample_id.
    predicted_label = int(predictions[name]['pred'][sample_id])  # Nhãn đã dự đoán.
    print(name, 'dự đoán:', CLASSES[predicted_label], '| thật:', CLASSES[sample_label])  # So sánh.''',
'Không yêu cầu hai mô hình đoán giống nhau. Với ảnh sai hoặc đúng, lời giải thích phải dựa vào bằng chứng thay vì gán ý nghĩa tùy ý cho feature map.')
lesson('33. Lưu kết quả và tải về',
'''ZIP gom nhiều file thành một gói. Notebook trên Drive giữ code và output hiển thị; checkpoint trong runtime có thể mất khi phiên kết thúc. Tải ZIP để giữ trọng số và số liệu. Chỉ có trạng thái hoàn thành sau khi các bước trước đã chạy đủ.''',
'''import shutil  # Công cụ đóng gói file.
from google.colab import files  # Tiện ích tải về của Colab.
save_json(RUN / 'STATUS.json', {'stage': 'completed_learning_notebook', 'epochs': EPOCHS, 'test_samples': 10000, 'limits': ['one seed', 'five epochs', 'no ablation', 'no exact resume']})  # Trạng thái thật.
archive_path = shutil.make_archive(str(RUN), 'zip', root_dir=RUN)  # Gói chỉ chứa kết quả run này.
print('Kết quả:', archive_path)  # In đường dẫn ZIP.
files.download(archive_path)  # Yêu cầu trình duyệt tải file về máy.''',
'Nếu tải bị chặn, mở Files bên trái, tìm file ZIP theo đường dẫn vừa in rồi Download. Không xóa bản cũ. Bản học tập chỉ lưu best checkpoint; không có tự resume chính xác.')
md('''## Sẵn sàng cho giai đoạn 3 khi nào?

- Giải thích được train/validation/test và chỉ train cập nhật trọng số.
- Đọc được [B,1,28,28], [B] và [B,10].
- Phân biệt pixel, trọng số, logits, loss, gradient và optimizer.
- Chỉ được dòng cập nhật tham số: optimizer.step().
- Biết cách sửa batch size mà không đổi số ảnh của split; hiểu 128 ở lớp MLP khác 128 ảnh/batch.
- Chọn mô hình bằng validation; không dùng test để chọn cấu hình.

**Bài tập nhỏ (làm trong ô mới):** tạo DataLoader(train_ds, batch_size=64, shuffle=False), lấy một batch, in shape. Đáp án: ảnh [64,1,28,28], nhãn [64], 844 batch cho 54k nếu giữ batch cuối. Không cần train lại.

**Thuật ngữ Python tra nhanh:** list `[]` = danh sách; tuple `(a,b)` = bộ; dictionary `{k:v}` = bảng khóa–giá trị; `None` = chưa có giá trị; `return` = trả kết quả; `for` = lặp; `with` = phạm vi chế độ; `self` = đối tượng hiện tại; `[:n]` = lấy trước vị trí n; `f'...'` = chuỗi chèn biến; `.to()` = chuyển thiết bị; `.item()` = lấy số đơn; `.detach()` = tách khỏi đồ thị; `.clone()` = sao chép; `.eval()` = chọn chế độ đánh giá.

**Xử lý lỗi:** NameError → chạy ô định nghĩa trước; lỗi CPU/CUDA → model và dữ liệu cùng DEVICE; backward twice → chạy lại từ bước 17; thiếu checkpoint → hoàn thành train trong cùng RUN; không import được → kiểm tra môi trường trước. Không coi outputs lưu từ lần trước là biến vẫn còn trong RAM.

Tài liệu tham khảo: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). Giữ lại kết quả xấu và giới hạn của thí nghiệm; không hứa trước accuracy.
''')

for i, cell in enumerate(cells):
    cell['id'] = f'hoc-cham-{i:03d}'
    cell['metadata']['id'] = cell['id']
notebook = dict(nbformat=4, nbformat_minor=5, metadata={
    'colab': {'name': 'Fashion_MNIST_Hoc_Cham_Giai_Thich_Tung_Dong.ipynb', 'provenance': []},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'language_info': {'name': 'python'}, 'accelerator': 'GPU'}, cells=cells)
target = HERE / 'Fashion_MNIST_Hoc_Cham_Giai_Thich_Tung_Dong.ipynb'
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
print(target)
print('Cells:', len(cells), 'Code:', sum(c['cell_type']=='code' for c in cells))
