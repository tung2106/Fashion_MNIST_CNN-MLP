# Notebook học chậm — bản riêng

[Mở bản gọn trên Colab](https://colab.research.google.com/drive/1hvzAjJsUWkK7WS5SVVZ1j1Bh1Ng7uTTS).

[Bản học chi tiết lưu trên Drive ngày 30/10/2026](https://colab.research.google.com/drive/1a0PONFuaDTd-3QD9saWxFUYarCQinwyo).

- `01_hoc_tung_buoc.ipynb`: 33 bước, 101 ô gồm giải thích, mỗi ô tập trung vào một thao tác và câu hỏi kiểm tra.
- [Script tạo bài học](../scripts/tao_bai_hoc.py): tạo lại bản học chậm cục bộ; không sửa bản Colab hoặc notebook gốc.
- [Script kiểm tra trên Colab GPU](../scripts/kiem_tra_bai_hoc.py): kiểm tra kỹ thuật với 128 ảnh train thật, không phải số liệu đánh giá đề tài.
- [Kết quả kiểm tra](../artifacts/kiem_tra_bai_hoc.json): phạm vi và kết quả kiểm tra.
- [Ảnh giao diện notebook](../../tai_lieu_du_an/huong_dan_colab/hinh/notebook_hoc_tung_buoc.png): ảnh xác nhận giao diện notebook mới.

## Cách học

1. Bước 1–12: Python, ảnh/pixel, nhãn, chia tập, chuẩn hóa và batch. Dừng sau bước 12 để tự giải thích kích thước tensor.
2. Bước 13–21: trọng số, gradient, MLP/CNN và từng thao tác học. Dừng sau bước 21.
3. Bước 22–27: đọc hàm train/validation rồi mới chạy huấn luyện. Bước 26 và 27 bắt đầu full train 5 epoch.
4. Bước 28–33: chốt model, test, phân tích và tải kết quả.

Mở bản Colab mới, đọc một phần rồi Shift+Enter từng ô. Không cần Run all. Kết quả mới được tạo tại `/content/fashion_mnist_hoc_cham/runs/<id>`, độc lập đường dẫn cũ.

## Đã kiểm tra / chưa chạy

Đã kiểm tra cú pháp 33 ô; smoke test MLP/CNN trên 128 mẫu thật xác nhận shape, train cập nhật trọng số và eval giữ nguyên trọng số. Bản kiểm tra local chỉ dùng ToTensor và không chạy toàn bộ pipeline Colab. Chưa chạy full 5 epoch trong notebook mới; không có accuracy mới được báo cáo. Giữ nguyên số liệu cũ ở thư mục run cũ.

Khác biệt phần phụ trợ: bản học chậm tách train và evaluate thành hai hàm, lưu best checkpoint và lịch sử; không có checkpoint last hoặc resume chính xác. Kiến trúc và cấu hình học cơ bản giữ như bản trước; không cam kết kết quả bit-for-bit giữa thiết bị/phiên bản.
