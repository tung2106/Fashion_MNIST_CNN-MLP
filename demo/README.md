# Demo phân loại ảnh thời trang

Demo chạy offline trên CPU với checkpoint đã huấn luyện trên Colab. Mỗi lượt nạp ba checkpoint, chạy dự đoán trên cùng một ảnh Fashion-MNIST rồi hiển thị nhãn thật, dự đoán của từng mô hình và top 3 của CNNPlus.

## Chạy ngay trên máy

Lệnh gọn để dự đoán và tự mở hình: `.\demo\chay_demo.ps1 -Mau 183`.
Đổi `183` thành `2914` để xem ví dụ sai. [Kịch bản trình bày](kich_ban_trinh_bay.md) có thứ tự demo 7 phút, câu hỏi thường gặp và cách chuẩn bị máy.

Mở terminal PowerShell ở thư mục gốc dự án và chạy từng cặp lệnh.

```powershell
.\.venv\Scripts\python.exe -m demo.so_sanh_mo_hinh --sample 183
Invoke-Item -LiteralPath 'demo/output/sample_183.png'

.\.venv\Scripts\python.exe -m demo.so_sanh_mo_hinh --sample 2914
Invoke-Item -LiteralPath 'demo/output/sample_2914.png'
```

Có thể đổi ID thành 19, 1, 6, 4 hoặc 3531. Lệnh đầu trong mỗi cặp chạy dự đoán thật và lưu hình. Lệnh sau mở hình để trình chiếu. Khi cần, giữ sẵn hai hình đã xuất để mở trực tiếp. Nếu chuyển sang máy khác, cần môi trường Python với các thư viện trong [requirements.txt](../requirements.txt) và ba checkpoint trong `models/checkpoints`.

[Notebook trình bày](../notebooks/06_demo_du_doan.ipynb) có các ô và hình kết quả theo đúng thứ tự bên dưới. Để chạy lại trong VS Code, chọn kernel Python của dự án có `ipykernel`.

## Trình bày trong 5–7 phút

| Thời lượng | Thao tác | Nội dung để nói |
|---|---|---|
| 45 giây | Mở ảnh #183 | Một ảnh xám 28×28 đi vào mô hình. Mô hình trả điểm cho 10 lớp và lấy lớp có điểm cao nhất. |
| 90 giây | Xem kết quả ảnh #183 | Nhãn thật là Shirt. MLP đoán T-shirt/top. Hai CNN đoán đúng Shirt. MLP trải phẳng ảnh thành 784 giá trị, còn các lớp tích chập học những mẫu ở các vùng lân cận bằng bộ lọc dùng chung. |
| 60 giây | Mở ảnh #2914 | Nhãn thật là Sneaker. CNNPlus đoán Ankle boot với điểm softmax khoảng 99,9954%. Điểm cao vẫn có thể đi kèm dự đoán sai. |
| 90 giây | Mở [bản đồ đặc trưng #19](../reports/results/20261104/features_19.png) | Chuẩn hóa là tiền xử lý. Các ô màu thể hiện kích hoạt do CNN học được, từ tầng đầu 28×28 đến tầng sâu 14×14. Chỉ ra vùng nào nổi lên hoặc thay đổi giữa hai tầng. |
| 90 giây | Mở [ma trận nhầm lẫn](../reports/results/20261104/CNNPlus_confusion.png) và [bảng kết quả](../reports/results/20261104/bao_cao_danh_gia.md) | Accuracy trên 10.000 ảnh là 88,55% với MLP, 92,11% với CNN cơ bản và 93,70% với CNNPlus. Có 94 ảnh Shirt bị đoán thành T-shirt/top. Hai ảnh demo được chọn để minh họa từng trường hợp. |

`examples/labels.json` ghi ID trong tập test, nhãn thật và kết quả gốc của CNNPlus. Lệnh dưới chỉ in top 3 dưới dạng văn bản. Cả hai lệnh dự đoán dùng chung `models.du_doan`.

```powershell
.\.venv\Scripts\python.exe -m demo.du_doan_anh --image demo/examples/sample_183.png
```
