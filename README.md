# Phân loại trang phục bằng CNN và MLP

Dự án dùng Fashion-MNIST để so sánh cách MLP và CNN phân loại cùng một ảnh thành 10 loại trang phục. Nếu mới học, mở [notebook 01 trên Colab](https://colab.research.google.com/drive/1hvzAjJsUWkK7WS5SVVZ1j1Bh1Ng7uTTS) hoặc [bản trong dự án](02_thuc_hanh_colab/notebooks/01_hoc_tung_buoc.ipynb).

| Giai đoạn | Mở tại đây | Nội dung |
|---|---|---|
| 01 · Khám phá dữ liệu | [xem_du_lieu.py](01_kham_pha_du_lieu/xem_du_lieu.py) | Xem ảnh 28×28, nhãn, pixel và số ảnh mỗi lớp. |
| 02 · Thực hành Colab | [Hướng dẫn](02_thuc_hanh_colab/README.md) | Học theo từng ô; so sánh MLP và CNN. Đã train và test bộ mô hình cơ bản. |
| 03 · Thực nghiệm | [Mã nguồn và lệnh chạy](03_thuc_nghiem/README.md) | Tách mã thành module. Đã có mốc 5 epoch trên Colab GPU; giai đoạn này mới đánh giá trên validation. |
| 04 · Demo | [Kế hoạch demo](04_demo/README.md) | Dùng trọng số đã học để dự đoán một ảnh. Chưa có ứng dụng riêng. |
| 05 · Bàn giao | [Danh mục bàn giao](05_ban_giao/README.md) | Chuẩn bị báo cáo, slide, hướng dẫn và gói nộp. Chưa có bản cuối. |

**Nơi lưu chung:** [du_lieu_chung](du_lieu_chung/README.md) chứa ảnh và nhãn gốc; [tai_lieu_du_an](tai_lieu_du_an/README.md) chứa hướng dẫn, ảnh minh họa và tài liệu nhóm. `luu_tru/` giữ tệp cũ trên máy.

**Đọc kết quả:** [bảng test giai đoạn 02](02_thuc_hanh_colab/artifacts/20261029_092231_60aa1b/metrics/summary.csv) ghi MLP 87,50%, CNN 90,27%. [Lần chạy GPU giai đoạn 03](https://colab.research.google.com/drive/1WiCTnJhuDdR6ovY-NCgWCHWbYDRqWdjH) ghi MLP 88,58%, CNN 91,32% trên **validation**; hai cặp số đo trên hai tập khác nhau.

**Quy ước:** dùng tên không dấu, chữ thường, nối từ bằng `_`. Chỉ đánh số khi thứ tự có ý nghĩa. Notebook ở `notebooks/`, script hỗ trợ ở `scripts/`, mã mô hình ở `src/`, kết quả ở `artifacts/<run_id>/` của giai đoạn tương ứng. Giữ nguyên tên file do thư viện và lần chạy sinh ra. Train trên Google Colab GPU.
