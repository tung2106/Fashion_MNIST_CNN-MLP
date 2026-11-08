# Kịch bản demo trước giảng viên

Đề tài: **Phân loại ảnh thời trang bằng CNN**. Giảng viên: **Dương Hiền Thanh**.

Mục tiêu trong 7 phút: cho thấy mô hình dự đoán thật, giải thích đường đi của ảnh và trình bày cả kết quả đúng lẫn sai. Demo chạy CPU từ checkpoint đã train trên Colab; không cần Internet hoặc chạy lại huấn luyện.

## Chuẩn bị trước buổi học

1. Mở đúng thư mục dự án trong VS Code. Giữ đủ ba checkpoint `mlp.pt`, `cnn.pt`, `cnn_plus.pt` trong `models/checkpoints/`; Git không chứa các tệp này.
2. Chạy kiểm tra bằng lệnh dưới. Kết quả cần có `inference_vs_original_predictions: PASS`.
3. Chạy thử hai ảnh 183 và 2914. Giữ hai hình trong `demo/output/` làm phương án dự phòng.
4. Mở sẵn báo cáo Word, hình đặc trưng và ma trận nhầm lẫn. Trên máy trình chiếu khác cần chép cả checkpoint, ảnh mẫu và môi trường Python phù hợp; thử trước ngày bảo vệ.

```powershell
.\.venv\Scripts\python.exe -m tests.kiem_tra_du_an
.\demo\chay_demo.ps1 -Mau 183
.\demo\chay_demo.ps1 -Mau 2914
```

Nếu PowerShell chặn chạy tệp `.ps1`, dùng trực tiếp hai lệnh dưới; không cần đổi chính sách máy:

```powershell
.\.venv\Scripts\python.exe -m demo.so_sanh_mo_hinh --sample 183
Invoke-Item -LiteralPath 'demo/output/sample_183.png'
```

Đổi `183` thành `2914` ở cả hai dòng để mở ví dụ sai. Để thử mà không mở cửa sổ ảnh, thêm `-KhongMoAnh` vào lệnh `chay_demo.ps1`.

## Trình tự trình bày

| Thời gian | Thao tác | Ý chính cần nói |
|---|---|---|
| 0:00–0:45 | Giới thiệu bài toán | Phân loại ảnh xám 28×28 thành 10 lớp. Fashion-MNIST có 70.000 ảnh: 54.000 train, 6.000 validation, 10.000 test. CNN là mô hình chính, MLP là đối chứng. |
| 0:45–2:15 | Chạy ảnh 183 và mở hình | Nhãn thật là Shirt. MLP đoán T-shirt/top; CNN và CNNPlus đoán đúng. MLP trải ảnh thành 784 số; CNN học các mẫu cục bộ bằng bộ lọc dùng chung. Nhãn thật chỉ dùng đối chiếu sau dự đoán. |
| 2:15–3:15 | Chạy ảnh 2914 | Đây là Sneaker, CNNPlus đoán Ankle boot với điểm softmax khoảng 99,9954%. Điểm cao không bảo đảm đúng. Hai ảnh demo là ví dụ minh họa, không đại diện cho accuracy toàn tập. |
| 3:15–4:30 | Mở `reports/results/20261104/features_19.png` và `features_2914.png` | Chuẩn hóa là tiền xử lý; bản đồ kích hoạt được tạo qua các bộ lọc CNN đã học. So sánh tầng đầu và tầng sâu, trên ảnh đúng và sai. Không gán chắc chắn mỗi kênh là “cổ áo” hay “đế giày”. |
| 4:30–5:45 | Mở ma trận nhầm lẫn và bảng test | MLP 88,55%; CNN 92,11%; CNNPlus 93,70%. CNNPlus đúng 9.370/10.000 ảnh. Shirt là lớp khó: recall 77,30%; 94 ảnh Shirt bị đoán thành T-shirt/top. |
| 5:45–7:00 | Chỉ đường đi trong mã và kết luận | `chuan_bi_du_lieu.py` → `kien_truc_mo_hinh.py` → `vong_lap_huan_luyen.py` → `danh_gia_mo_hinh.py`. Khi demo, `du_doan.py` nạp trọng số và dùng lại chuẩn hóa. Giới hạn: ảnh điện thoại chưa được đánh giá; mô hình vẫn sai; không tuyên bố tối ưu tuyệt đối. |

## Những câu cần giải thích được

**Train, validation và test khác nhau thế nào?** Train cập nhật trọng số. Validation chọn checkpoint và cấu hình. Test báo cáo chất lượng cấu hình đã chốt. Không chỉnh mô hình dựa trên kết quả test.

**Vì sao CNN thường phù hợp với ảnh hơn MLP trong bài này?** CNN khai thác vùng lân cận và dùng cùng bộ lọc ở nhiều vị trí. MLP vẫn giữ giá trị pixel khi flatten nhưng không có cấu trúc tích chập đó. Kết quả của nhóm cho thấy CNN tốt hơn trong thiết lập đã thử.

**Logits và loss là gì?** Logits là 10 điểm đầu ra cho 10 lớp. Cross-Entropy đo mức không phù hợp giữa các điểm này và nhãn đúng; không so sánh từng pixel. Trong train, gradient được tính từ loss và Adam cập nhật trọng số.

**Vì sao có ba con số epoch khác nhau?** Epoch tối đa là ngân sách; epoch đã chạy có thể ít hơn do early stopping; epoch checkpoint là lần có validation loss thấp nhất. Chúng không nhất thiết bằng nhau.

**Chọn mô hình bằng loss hay F1?** Trong mỗi lượt train chọn checkpoint theo validation loss nhỏ nhất. Giữa các cấu hình chọn theo macro-F1 validation cao nhất, hòa thì chọn loss thấp hơn.

**CNNPlus có phải chỉ thêm Dropout không?** Không. CNNPlus thay nhiều yếu tố: bốn convolution, BatchNorm, Dropout và số kênh. Tác động riêng của từng kỹ thuật được xem qua các đối chứng một yếu tố trên CNN gốc.

**Augmentation có luôn tốt không?** Không. Với seed và ngân sách đã thử, augmentation riêng trên CNN cơ bản không tăng macro-F1 validation. Nó giúp khi kết hợp CNNPlus. Không suy rộng từ một thí nghiệm thành quy luật.

**Vì sao demo dùng seed 42 dù seed khác có test cao hơn?** Seed demo được định trước. Chọn seed bằng test sẽ biến test thành dữ liệu lựa chọn.

**Feature maps có giải thích chính xác vì sao dự đoán sai không?** Chúng cho thấy các kích hoạt và gợi ý để phân tích. Chưa đủ chứng minh một kênh cụ thể gây lỗi hoặc một vùng ảnh có ý nghĩa ngữ nghĩa cố định.

**Demo có học thêm mỗi khi bấm chạy không?** Không. Nó dùng `eval()` và `inference_mode()`, không có optimizer hoặc bước cập nhật trọng số. Chạy lại chỉ tính dự đoán từ trọng số đã lưu.

## Khi có sự cố

- Nếu thiếu checkpoint: bổ sung đúng ba tệp đã kiểm chứng; không train ngay trong buổi demo.
- Nếu thiếu thư viện: dùng máy đã chuẩn bị và kiểm tra trước. Môi trường tham khảo ghi ở `training/configs/moi_truong.json`.
- Nếu không chạy được trực tiếp: mở hình đã xuất và nói rõ đây là kết quả đã chạy trước đó. Không trình bày hình lưu sẵn như một lượt dự đoán vừa chạy.
- Nếu ảnh đầu vào bị từ chối: demo hiện nhận ảnh xám 28×28, không nhận trực tiếp ảnh màu điện thoại.

Nguồn số liệu: [bảng kết quả và phân tích](../reports/results/20261104/bao_cao_danh_gia.md). Mỗi thành viên nên tự chạy hai ví dụ và giải thích được luồng dữ liệu trước khi trình bày.
