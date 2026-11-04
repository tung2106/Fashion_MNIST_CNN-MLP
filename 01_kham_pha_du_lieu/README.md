# Giai đoạn 01 — Làm quen dữ liệu

**Hiện có:** [xem_du_lieu.py](xem_du_lieu.py), chính là `test.py` trước đây. Mã đọc Fashion-MNIST từ thư mục chung `../du_lieu_chung`, hiển thị ảnh, nhãn, kích thước tensor, vùng pixel và số ảnh mỗi lớp.

Chạy từ terminal tại gốc dự án, trong môi trường đã có torch, torchvision và matplotlib:

```powershell
python "01_kham_pha_du_lieu/xem_du_lieu.py"
```

`plt.show()` có thể giữ tiến trình chạy đến khi đóng cửa sổ ảnh. Bài này chưa huấn luyện mô hình.

Nếu cửa sổ không hiện, mở `anh_dang_xem.png` ngay trong thư mục này sau khi chạy thành công. Script lưu ảnh trước khi gọi `plt.show()`.

Nếu Python chưa có torch/matplotlib, có thể xuất cùng ảnh `dataset[1]` bằng Python chuẩn:

```powershell
python "01_kham_pha_du_lieu/xuat_anh_khong_can_thu_vien.py"
```

Mở [ảnh dữ liệu số 1](anh_du_lieu_1.png) để xem. Ảnh được phóng to từ 28×28 lên 280×280, giữ nguyên từng pixel.

Trước khi sang giai đoạn 02, giải thích được: ảnh xám 28×28; tensor `[1,28,28]`; nhãn 0–9; pixel sau ToTensor ở khoảng 0–1; 60k train gốc tách thành 54k train và 6k validation, 10k test riêng.

Lưu các bài tập nhỏ về dữ liệu tại đây. Hướng dẫn dài/bài ghi chú chung đưa vào `tai_lieu_du_an`. [Về bản đồ dự án](../README.md).
