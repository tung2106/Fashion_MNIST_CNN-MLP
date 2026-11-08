# Notebook

| Tệp | Mục đích |
|---|---|
| `01_kien_thuc_co_ban.ipynb` | Học Python/PyTorch theo ô nhỏ |
| `02_thuc_hanh_mlp_cnn.ipynb` | Quy trình cơ bản MLP/CNN |
| `03_chay_ma_nguon.ipynb` | Gọi mã dùng chung, có inspect và smoke |
| `04_huan_luyen_colab.ipynb` | Train baseline từ gói mã |
| `05_chon_mo_hinh.ipynb` | Giao thức lựa chọn mô hình và đánh giá |
| `06_demo_du_doan.ipynb` | Demo offline trước giảng viên, xem dự đoán, đặc trưng CNN và kết quả test |

Số thứ tự là thứ tự đọc notebook, không chia dự án thành giai đoạn. Mọi ô cập nhật trọng số chỉ chạy Colab GPU. Bản local đã chỉnh đường dẫn và kiểm tra cú pháp; chưa thực thi train lại sau khi tổ chức thư mục.

Ở máy: `python -m scripts.dong_goi_colab` → upload `experiments/source.zip` qua Files của Colab → chọn T4 → mở notebook 04 hoặc 05. Notebook 03 cần PROJECT trỏ tới mã đã giải nén. Notebook 01/02 tự chứa bài học.

Notebook Drive đã chạy trước đây giữ nguyên làm bằng chứng: [bài học](https://colab.research.google.com/drive/1hvzAjJsUWkK7WS5SVVZ1j1Bh1Ng7uTTS), [baseline](https://colab.research.google.com/drive/1WiCTnJhuDdR6ovY-NCgWCHWbYDRqWdjH), [lựa chọn mô hình](https://colab.research.google.com/drive/1K9FNgM9mzUL55AmOaVcc_vIfLd5DKkP0). Đường dẫn trong output lịch sử thuộc phiên chạy cũ.
