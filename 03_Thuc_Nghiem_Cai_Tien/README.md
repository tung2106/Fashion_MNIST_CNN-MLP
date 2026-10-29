# Giai đoạn 03 — Thí nghiệm cải tiến

**Trạng thái: chưa triển khai.** Đây là nơi xây mã huấn luyện/đánh giá dùng chung và thực hiện thí nghiệm chính của đồ án.

Công việc theo [đặc tả](../TAI_LIEU/01_Dac_Ta_Va_Ke_Hoach/01_DAC_TA_VA_PROMPT.md):

- Thiết lập split, seed, tiền xử lý và tiêu chí chọn checkpoint trên validation.
- So sánh MLP/CNN và các thí nghiệm cốt lõi E2–E6; quyết định E7 theo kết quả validation.
- Phân tích đặc trưng học được, ảnh sai, các lớp dễ nhầm; giải thích tác động của từng thay đổi để đáp ứng phần nhận xét về feature engineering.
- Lưu cấu hình, thời gian, tham số, checkpoint, dự đoán và chỉ số cho từng run. Không dùng test để lựa chọn cấu hình.

Cấu trúc sẽ tạo khi triển khai: `src/` (mã dùng chung), `configs/` (cấu hình), `scripts/` (lệnh chạy), `notebooks/` (thí nghiệm), `tests/` (kiểm tra cần thiết), `artifacts/<run_id>/` (kết quả). Đọc dữ liệu từ `../DU_LIEU`; tài liệu nội dung nằm trong `../TAI_LIEU`.

HOG/SVM, nhiều seed và thử độ bền là mở rộng có điều kiện trong đặc tả. Không xem notebook cơ bản giai đoạn 02 là đã hoàn thành các việc trên. [Về bản đồ dự án](../README.md).
