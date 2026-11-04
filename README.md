# Phân loại trang phục bằng CNN và MLP

Dự án dùng Fashion-MNIST để so sánh cách MLP và CNN phân loại cùng một ảnh thành 10 loại trang phục. Nếu mới học, mở [notebook 01 trên Colab](https://colab.research.google.com/drive/1hvzAjJsUWkK7WS5SVVZ1j1Bh1Ng7uTTS) hoặc [bản trong dự án](02_thuc_hanh_colab/notebooks/01_hoc_tung_buoc.ipynb).

[Đánh giá mô hình để chuẩn bị demo](tai_lieu_du_an/bao_cao_nhom/ket_qua_20261104/danh_gia_chuan_bi_demo.md): kết quả đã tải và kiểm chứng, phân tích đặc trưng/lỗi, giới hạn và phần việc còn lại.

| Giai đoạn | Mở tại đây | Nội dung |
|---|---|---|
| 01 · Khám phá dữ liệu | [xem_du_lieu.py](01_kham_pha_du_lieu/xem_du_lieu.py) | Xem ảnh 28×28, nhãn, pixel và số ảnh mỗi lớp. |
| 02 · Thực hành Colab | [Hướng dẫn](02_thuc_hanh_colab/README.md) | Học theo từng ô; so sánh MLP và CNN. Đã train và test bộ mô hình cơ bản. |
| 03 · Thực nghiệm | [Mã nguồn và lệnh chạy](03_thuc_nghiem/README.md) | Đã so sánh 7 cấu hình trên Colab GPU, xác nhận cấu hình chọn qua 3 seed và đánh giá test. |
| 04 · Demo | [Kế hoạch demo](04_demo/README.md) | Đã có mã inference dùng checkpoint. Chưa có giao diện riêng. |
| 05 · Bàn giao | [Danh mục bàn giao](05_ban_giao/README.md) | Chuẩn bị báo cáo, slide, hướng dẫn và gói nộp. Chưa có bản cuối. |

**Nơi lưu chung:** [du_lieu_chung](du_lieu_chung/README.md) chứa ảnh và nhãn gốc; [tai_lieu_du_an](tai_lieu_du_an/README.md) chứa hướng dẫn, ảnh minh họa và tài liệu nhóm. `luu_tru/` giữ tệp cũ trên máy.

**Kết quả mới 04/11/2026:** [notebook lựa chọn mô hình](https://colab.research.google.com/drive/1K9FNgM9mzUL55AmOaVcc_vIfLd5DKkP0) ghi accuracy **test** MLP 88,55%, CNN cơ bản 92,11%, CNNPlus + augmentation 93,70%. Cấu hình được chọn bằng validation, không bằng test. Trung bình test qua 3 seed: 93,80% ± 0,091 điểm phần trăm (độ lệch chuẩn mẫu). Xem [tiến độ và trạng thái tải kết quả](tai_lieu_du_an/huong_dan_thuc_nghiem/tien_do_giai_doan_03.md).

**Mốc lịch sử:** [test giai đoạn 02](02_thuc_hanh_colab/artifacts/20261029_092231_60aa1b/metrics/summary.csv) MLP 87,50%, CNN 90,27%; [mốc 5 epoch giai đoạn 03](https://colab.research.google.com/drive/1WiCTnJhuDdR6ovY-NCgWCHWbYDRqWdjH) MLP 88,58%, CNN 91,32% trên **validation**. Không trộn số đo validation với test.

**Quy ước:** dùng tên không dấu, chữ thường, nối từ bằng `_`. Chỉ đánh số khi thứ tự có ý nghĩa. Notebook ở `notebooks/`, script hỗ trợ ở `scripts/`, mã mô hình ở `src/`, kết quả ở `artifacts/<run_id>/` của giai đoạn tương ứng. Giữ nguyên tên file do thư viện và lần chạy sinh ra. Train trên Google Colab GPU.
