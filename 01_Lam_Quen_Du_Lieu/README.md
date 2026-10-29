# Giai đoạn 01 — Làm quen dữ liệu

**Hiện có:** [xem_du_lieu.py](xem_du_lieu.py), chính là `test.py` trước đây. Mã đọc Fashion-MNIST từ thư mục chung `../DU_LIEU`, hiển thị ảnh, nhãn, kích thước tensor, vùng pixel và số ảnh mỗi lớp.

Chạy từ terminal tại gốc dự án, trong môi trường đã có torch, torchvision và matplotlib:

```powershell
python "01_Lam_Quen_Du_Lieu/xem_du_lieu.py"
```

`plt.show()` có thể giữ tiến trình chạy đến khi đóng cửa sổ ảnh. Bài này chưa huấn luyện mô hình.

Trước khi sang giai đoạn 02, giải thích được: ảnh xám 28×28; tensor `[1,28,28]`; nhãn 0–9; pixel sau ToTensor ở khoảng 0–1; 60k train gốc tách thành 54k train và 6k validation, 10k test riêng.

Lưu các bài tập nhỏ về dữ liệu tại đây. Hướng dẫn dài/bài ghi chú chung đưa vào `TAI_LIEU`. [Về bản đồ dự án](../README.md).
