# Notebook học chậm — bản riêng

[Mở notebook mới trên Colab](https://colab.research.google.com/drive/1a0PONFuaDTd-3QD9saWxFUYarCQinwyo).

Đã nhập và xác nhận lưu trên Google Drive ngày 30/10/2026. Notebook gốc và kết quả cũ không bị sửa.

- `Fashion_MNIST_Hoc_Cham_Giai_Thich_Tung_Dong.ipynb`: 33 bước, 101 ô gồm giải thích, code có chú thích từng dòng và câu hỏi kiểm tra.
- `build_hoc_cham.py`: tạo lại bản học chậm cục bộ; không sửa bản Colab hoặc notebook gốc.
- `verify_hoc_cham.py`: kiểm tra kỹ thuật với 128 ảnh train thật, không phải số liệu đánh giá đề tài.
- `KIEM_TRA_NOTEBOOK.json`: phạm vi và kết quả kiểm tra.
- `colab_notebook_moi.png`: ảnh xác nhận giao diện notebook mới.

## Cách học

1. Bước 1–12: Python, ảnh/pixel, nhãn, chia tập, chuẩn hóa và batch. Dừng sau bước 12 để tự giải thích kích thước tensor.
2. Bước 13–21: trọng số, gradient, MLP/CNN và từng thao tác học. Dừng sau bước 21.
3. Bước 22–27: đọc hàm train/validation rồi mới chạy huấn luyện. Bước 26 và 27 bắt đầu full train 5 epoch.
4. Bước 28–33: chốt model, test, phân tích và tải kết quả.

Mở bản Colab mới, đọc một phần rồi Shift+Enter từng ô. Không cần Run all. Kết quả mới được tạo tại `/content/fashion_mnist_hoc_cham/runs/<id>`, độc lập đường dẫn cũ.

## Đã kiểm tra / chưa chạy

Đã kiểm tra cú pháp 33 ô; smoke test MLP/CNN trên 128 mẫu thật xác nhận shape, train cập nhật trọng số và eval giữ nguyên trọng số. Bản kiểm tra local chỉ dùng ToTensor và không chạy toàn bộ pipeline Colab. Chưa chạy full 5 epoch trong notebook mới; không có accuracy mới được báo cáo. Giữ nguyên số liệu cũ ở thư mục run cũ.

Khác biệt phần phụ trợ: bản học chậm tách train và evaluate thành hai hàm, lưu best checkpoint và lịch sử; không có checkpoint last hoặc resume chính xác. Kiến trúc và cấu hình học cơ bản giữ như bản trước; không cam kết kết quả bit-for-bit giữa thiết bị/phiên bản.
