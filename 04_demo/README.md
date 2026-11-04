# Giai đoạn 04 — Demo trực quan

**Trạng thái: mã inference đã chạy thành công với checkpoint đã tải; chưa xây giao diện riêng.** Chỉ nạp trọng số đã chọn ở giai đoạn 3, không train khi demo. Xem [đánh giá và bằng chứng](../tai_lieu_du_an/bao_cao_nhom/ket_qua_20261104/danh_gia_chuan_bi_demo.md).

`predict.py` nhận một checkpoint và ảnh xám 28×28, trả top-3 dự đoán. Tái sử dụng `03_thuc_nghiem/src/inference.py` để giữ preprocessing giống đánh giá. Chạy từ gốc dự án:

```powershell
$run = '.\03_thuc_nghiem\artifacts\selection_20261104_144952_e58647'
.\.venv\Scripts\python.exe .\04_demo\predict.py --checkpoint "$run\demo_bundle\model.pt" --image "$run\analysis\sample_19.png"
```

Ảnh ID 19 được đoán đúng là T-shirt/top (99,79%). Đổi thành `sample_2914.png` để xem trường hợp sai: nhãn Sneaker nhưng đoán Ankle boot, dù điểm softmax rất cao. Đã đối chiếu 6 ảnh với Colab, khớp trong dung sai. Lệnh chỉ inference trên CPU, không train; có thể chạy inference trên Colab. Điểm softmax không phải độ tin cậy đã hiệu chuẩn. Trọng số/ảnh nằm trong artifact local, không tự có khi clone GitHub.

Nơi lưu mã giao diện và tài nguyên demo. Công việc: chọn ảnh Fashion-MNIST, hiển thị ảnh đầu vào, dự đoán, phân bố điểm/xác suất và hình minh họa đặc trưng; chỉ ra trường hợp mô hình nhầm. Upload ảnh ngoài là mở rộng, cần giải thích dữ liệu ngoài có thể khác Fashion-MNIST.

Tái sử dụng mô hình và tiền xử lý của giai đoạn 03. Nạp checkpoint đã chọn; không huấn luyện lại mỗi lần người dùng bấm dự đoán. Thông tin mô hình phải chỉ tới đúng run nguồn. Hướng dẫn sử dụng và ảnh chụp phục vụ báo cáo đặt trong `../tai_lieu_du_an/bao_cao_nhom`.

[Tài liệu dự án](../tai_lieu_du_an/README.md) · [Về bản đồ dự án](../README.md).
