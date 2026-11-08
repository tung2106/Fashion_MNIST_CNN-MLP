# Kiểm tra dự án và mô hình ngày 08 tháng 11 năm 2026

Ba checkpoint MLP, CNN và CNNPlus nạp được và chạy dự đoán trên CPU. Đã tính lại dự đoán cho **10.000 ảnh test trên từng mô hình**, tổng cộng 30.000 lượt dự đoán. Toàn bộ nhãn dự đoán khớp kết quả Colab đã lưu. Có thể dùng các checkpoint hiện tại để demo Fashion-MNIST.

Đây là kiểm tra hoạt động sau khi sắp xếp và format mã nguồn. Không huấn luyện lại, không cập nhật trọng số và không dùng kết quả test để chọn mô hình mới.

## Kết quả suy luận được kiểm tra lại

| Checkpoint | Số ảnh đúng trên 10.000 | Accuracy | Macro-F1 | Tham số |
|---|---:|---:|---:|---:|
| MLP | 8.855 | 88,55% | 88,55% | 101.770 |
| CNN cơ bản | 9.211 | 92,11% | 92,09% | 206.922 |
| CNNPlus đã chọn | 9.370 | 93,70% | 93,66% | 468.010 |

Nguồn là run `selection_20261104_144952_e58647`, bản CNNPlus dùng seed 42. Sai lệch xác suất lớn nhất giữa suy luận CPU lần này và bảng Colab là khoảng `4,752 × 10^-6`, trong dung sai đối chiếu `atol=2e-5, rtol=2e-4`. Nhãn dự đoán không có khác biệt.

## Các kiểm tra đã chạy trong lần này

| Phần kiểm tra | Kết quả |
|---|---|
| Ruff lint và định dạng mã hiện hành | PASS |
| Cú pháp Python và code cell của 6 notebook | PASS |
| Import các module chính | PASS |
| Guard từ chối huấn luyện ngoài Colab GPU | PASS |
| 60.000 ảnh gốc train và 10.000 ảnh test | Đủ số lượng |
| SHA256 của 8 tệp dữ liệu gốc | Khớp manifest đã lưu |
| Chia 54.000 train và 6.000 validation với seed 42 | Không giao nhau; khớp ID và thứ tự của cả 9 lượt thực nghiệm |
| Số ảnh mỗi lớp | Train 5.400; validation 600; test 1.000 |
| Mean và std tính lại chỉ trên tập train | Khớp 0,2862090170 và 0,3531597555 |
| Epoch checkpoint tốt nhất | Khớp epoch có validation loss thấp nhất trong lịch sử của cả 9 lượt |
| Quy tắc chọn cấu hình | CNNPlus + augmentation đứng đầu macro-F1 validation của 7 cấu hình seed 42 |
| 5 bảng dự đoán test đã lưu | Accuracy, macro-F1, classification report, confusion matrix và support đều khớp tính lại |
| Trung bình và độ lệch chuẩn 3 seed | Khớp stability.json đã lưu |
| 6 bảng/tệp nguồn đưa vào báo cáo | Hash khớp artifact gốc |
| 3 checkpoint demo | Hash khớp registry và checkpoint gốc; trạng thái không đổi sau dự đoán |
| 7 ảnh demo × 3 mô hình | 21 dự đoán từng ảnh khớp xác suất Colab |
| 10.000 ảnh test × 3 mô hình | 30.000 nhãn dự đoán khớp; xác suất khớp trong dung sai |
| 5 lớp kiến trúc mô hình | Đầu ra `[2, 10]`, giá trị hữu hạn khi suy luận với batch thử |
| Ảnh RGB hoặc ảnh khác 28×28 | Bị từ chối bằng ValueError như thiết kế |
| Cùng một ảnh dự đoán hai lần | Kết quả giống nhau trong chế độ eval |
| 177 tệp artifact và checkpoint được băm trước/sau | Không đổi nội dung |
| `git diff --check` | Không có lỗi khoảng trắng; Git chỉ nhắc chuẩn hóa LF/CRLF |

Các kiểm tra dùng batch giả chỉ kiểm tra đầu ra của kiến trúc, không được dùng làm kết quả chất lượng mô hình. Kết quả accuracy trong bảng đầu được tính từ tập test Fashion-MNIST thật.

## Lệnh đã thực hiện

Chạy từ thư mục gốc dự án, với môi trường `.venv` hiện có:

```powershell
.\.venv\Scripts\ruff.exe check data models training evaluation demo scripts utils tests notebooks
.\.venv\Scripts\ruff.exe format --check data models training evaluation demo scripts utils tests notebooks
.\.venv\Scripts\python.exe -m tests.kiem_tra_du_an
.\.venv\Scripts\python.exe -m tests.kiem_tra_mo_hinh --output reports/verification/20261108/kiem_tra_mo_hinh.json
git diff --check
```

Chi tiết máy đọc được: [kiem_tra_mo_hinh.json](kiem_tra_mo_hinh.json). Mã kiểm tra: [tests/kiem_tra_mo_hinh.py](../../../tests/kiem_tra_mo_hinh.py).

## Kiểm tra demo để trình bày

- Hai lệnh tạo hình với ID 183 và 2914 đã chạy thành công; đã mở kiểm tra bố cục cả hai hình.
- Lệnh dự đoán một ảnh trả top 3 của ảnh 183 theo thứ tự Shirt, T-shirt/top, Dress.
- `demo/chay_demo.ps1 -Mau 183 -KhongMoAnh` đã chạy thành công. Chế độ mặc định mở hình sau khi dự đoán để tiện trình chiếu.
- Đã thực thi lần lượt 5 ô Python của notebook demo 06 bằng môi trường Python hiện có và Matplotlib Agg. Không chạy kernel Jupyter hoặc cập nhật output đã lưu; phần hiển thị được đối chiếu qua các PNG.
- ID ảnh không tồn tại được CLI từ chối với mã thoát 2.

Chi tiết: [demo_verification.json](demo_verification.json). Kịch bản và câu hỏi bảo vệ: [hướng dẫn trình bày](../../../demo/kich_ban_trinh_bay.md).

## Phạm vi kết luận

- Các checkpoint hoạt động đúng với ảnh xám 28×28 thuộc Fashion-MNIST. Chưa kiểm chứng chất lượng với ảnh quần áo chụp từ điện thoại.
- CNNPlus vẫn đoán sai 630 ảnh test. Điểm softmax cao không bảo đảm đúng; demo cần có cả ví dụ sai.
- Bộ test đã được xem ở giai đoạn trước. Lần chạy này là kiểm tra tái lập kết quả, không phải đánh giá trên một bộ dữ liệu hoàn toàn mới.
- Không chạy lại các test có optimizer, backpropagation, scheduler hoặc early stopping trên máy cá nhân. Những kiểm tra huấn luyện đã chạy trên Colab GPU ngày 04/11/2026; lần này chỉ đọc lịch sử và kiểm tra đường suy luận.
- Không phát hiện lỗi chặn demo trong các phép kiểm tra đã thực hiện. Điều này không phải bảo đảm phần mềm không còn lỗi ở mọi đầu vào và mọi môi trường.
