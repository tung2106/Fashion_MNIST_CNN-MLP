# Phân loại ảnh thời trang bằng CNN trên Fashion-MNIST

Dự án học sâu phân loại ảnh xám 28×28 thành 10 lớp. CNN là mô hình chính; MLP làm đối chứng. Mã được tổ chức theo chức năng, dùng chung cho huấn luyện, đánh giá và demo.

| Thư mục | Có gì? | Mở khi cần |
|---|---|---|
| [models](models/README.md) | Kiến trúc MLP/CNN, nạp mô hình, checkpoint đã chọn | Xem mô hình, dùng trọng số cho demo |
| [data](data/README.md) | Dữ liệu gốc, chia tập, chuẩn hóa, augmentation | Hiểu đầu vào của mô hình |
| [training](training/README.md) | Vòng học, cấu hình, lựa chọn mô hình | Train trên Colab GPU |
| [evaluation](evaluation/README.md) | Tính chỉ số, đường học, ma trận nhầm và đặc trưng | Phân tích chất lượng/lỗi |
| [demo](demo/README.md) | Ảnh mẫu và lệnh tạo hình dự đoán offline | Trình bày demo |
| [reports](reports/README.md) | Báo cáo, bảng và hình đã kiểm chứng | Viết báo cáo, slide |
| [notebooks](notebooks/README.md) | Notebook học tập và chạy Colab | Thực hành từng bước |
| [docs](docs/README.md) | Kiến thức nền, hướng dẫn, nguồn tham khảo | Tra cứu |
| [experiments](experiments/README.md) | Các lần chạy gốc, log, ZIP và source snapshot | Truy vết số liệu |
| `scripts/`, `tests/`, `utils/` | Lệnh hỗ trợ, kiểm tra và đường dẫn chung | Bảo trì dự án |
| `archive/` | Bản cũ giữ trên máy để đối chiếu | Xem lịch sử, không chạy như mã hiện hành |

## Bắt đầu

Từ thư mục gốc dự án, dùng Python trong môi trường `.venv` đã có:

```powershell
.\.venv\Scripts\python.exe -m demo.du_doan_anh --image demo/examples/sample_19.png
```

Lệnh nạp `models/checkpoints/cnn_plus.pt`, trả top-3. Đổi ảnh thành `sample_2914.png` để xem ví dụ đoán sai. Inference có thể chạy CPU; mọi train/smoke/kiểm tra cập nhật trọng số chỉ chạy Colab GPU.

```powershell
# Chỉ đóng gói mã để upload Colab, không train:
.\.venv\Scripts\python.exe -m scripts.dong_goi_colab
```

Upload `experiments/source.zip` rồi dùng [notebook lựa chọn mô hình](notebooks/05_chon_mo_hinh.ipynb). Thư viện khai báo tại [requirements.txt](requirements.txt); không cần cài lại nếu môi trường hiện có vẫn hoạt động.

## Kết quả đã có

Run `selection_20261104_144952_e58647`, train Tesla T4 trên Colab ngày 04/11/2026:

| Mô hình, seed 42 | Accuracy test |
|---|---:|
| MLP | 88,55% |
| CNN cơ bản | 92,11% |
| CNNPlus + augmentation | **93,70%** |

Cấu hình được chọn bằng validation; 3 seed có accuracy test trung bình 93,80%, độ lệch chuẩn mẫu 0,091 điểm phần trăm. Đọc [báo cáo đánh giá](reports/results/20261104/bao_cao_danh_gia.md). Giữ nguyên split, mô hình và trọng số khi đổi cấu trúc thư mục.

## Chuẩn bị demo và báo cáo

Mở [báo cáo Word đầy đủ](reports/bao_cao_phan_loai_thoi_trang_hoan_chinh.docx) và [kịch bản demo 7 phút](demo/kich_ban_trinh_bay.md). Lệnh `.\demo\chay_demo.ps1 -Mau 183` chạy dự đoán CPU và mở hình; đổi thành `2914` để trình bày ví dụ sai. [Hướng dẫn demo offline](demo/README.md) có lệnh Python thay thế. [Notebook demo](notebooks/06_demo_du_doan.ipynb) có phần so sánh ba mô hình, ảnh dự đoán sai, bản đồ đặc trưng và ma trận nhầm lẫn.

[Kiểm tra ngày 08/11/2026](reports/verification/20261108/bien_ban_kiem_tra.md): đã chạy lại 10.000 ảnh test cho mỗi checkpoint MLP/CNN/CNNPlus trên CPU; cả 30.000 nhãn dự đoán khớp kết quả Colab. Kiểm tra mã, chia dữ liệu, chuẩn hóa, bảng chỉ số và checkpoint đều đạt. Đây là kiểm tra suy luận với trọng số đã lưu, không phải lượt huấn luyện mới.

Ba checkpoint demo và ảnh mẫu có sẵn trong GitHub; xem [cài đặt trên máy mới](demo/README.md#cài-trên-máy-mới). ZIP, dữ liệu gốc, các trọng số thực nghiệm khác và hướng dẫn AI giữ local theo `.gitignore`. Các notebook đã lưu trên Drive là bằng chứng lịch sử. Bản mã hiện hành dùng cấu trúc thư mục mới và demo offline; các notebook huấn luyện sau sắp xếp chưa chạy lại trên Colab.
