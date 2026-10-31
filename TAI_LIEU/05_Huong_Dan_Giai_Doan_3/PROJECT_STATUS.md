# Trạng thái giai đoạn 3 — 31/10/2026

## Đã thực hiện

- Tách mã cơ bản từ notebook giai đoạn 2 thành data/models/engine/evaluate/plots/utils và main.py.
- Tạo cấu hình baseline, smoke; lệnh inspect không train, train mặc định không xem test.
- Tạo môi trường `.venv`, dùng lại torch/torchvision có sẵn; các gói bổ sung cài vào venv. Phiên bản đã kiểm tra được ghi tại `03_Thuc_Nghiem_Cai_Tien/environment_verified.json`.
- Viết hướng dẫn tiếng Việt và notebook mỏng gọi code chung. Notebook mỏng đã kiểm tra cú pháp, chưa upload hoặc chạy trên Colab.
- Không sửa notebook/mã cũ và không tác động phiên GPU Colab đang được giữ lại.

## Kiểm tra thực tế

| Kiểm tra | Kết quả |
|---|---|
| 6 kiểm tra tự động bằng unittest | PASS |
| Split cân bằng, không giao nhau, cùng seed tái tạo được | PASS |
| Split trên dữ liệu thật so với splits.npz run cũ | Giống nhau từng ID và thứ tự |
| MLP/CNN mới so với hai class notebook, dùng cùng trọng số | Đầu ra giống chính xác trên tensor kiểm tra |
| Train sửa trọng số, validation giữ nguyên | PASS cho MLP và CNN |
| Loss gộp theo số ảnh, kể cả batch cuối nhỏ | PASS |
| Thứ tự dự đoán, tổng xác suất và kích thước đầu ra | PASS |
| Checkpoint → dự đoán → CSV/ma trận/accuracy/hash | PASS trên tensor giả trong thư mục test tạm; không phải kết quả Fashion-MNIST |
| Cú pháp các file Python và code cells notebook mỏng | PASS |
| git diff --check | PASS |

**Chạy thử với dữ liệu thật (smoke):**
`03_Thuc_Nghiem_Cai_Tien/artifacts/20261031_105329_eed4fb16`.
320 ảnh train / 80 validation; mỗi mô hình 1 epoch; CPU. MLP và CNN đều hoàn thành, có best/last checkpoint, history và đường cong. `STATUS.json` ghi completed, profile smoke, test_evaluated false. Không dùng các chỉ số run này làm số liệu chất lượng đồ án.

**Xem dữ liệu đầy đủ (inspect):**
`03_Thuc_Nghiem_Cai_Tien/artifacts/20261031_105521_a4175a5a`.
54.000 train / 6.000 validation, batch ảnh [128,1,28,28], nhãn [128]. Có splits, preprocessing, config và ảnh train_samples.png đã kiểm tra trực quan. Không train và không xem test.

Artifacts giữ trên máy và được loại khỏi Git mặc định. Các đường dẫn trên tính từ gốc dự án; nếu clone repository ở máy khác, chạy lại lệnh để tạo artifact mới. Tài liệu này ghi lại bằng chứng đã chạy trên máy hiện tại, không giả định các file local đi kèm repository.

## Chưa thực hiện ở giai đoạn 3

- Huấn luyện đầy đủ 5 epoch trên 54k ảnh và đánh giá test 10k của mã module mới.
- Ablation Dropout, BatchNorm, augmentation; nhiều seed; phân tích đặc trưng/lỗi sâu.
- Chức năng resume chính xác, evaluate-only và ứng dụng demo.
- Push thay đổi giai đoạn 3 lên GitHub. Notebook học chậm của giai đoạn 2 đã được push ở bước trước.

Giữ tiêu chí min validation loss để khớp notebook học tập. Khi chuyển sang thí nghiệm cải tiến với tiêu chí macro-F1 theo đặc tả, cần ghi giao thức mới và chạy đối chứng cùng tiêu chí. Không dùng test để chọn cấu hình.

## Bước học tiếp theo

Đọc `HUONG_DAN_DOC_CODE.md` phần 1–3 rồi chạy `main.py --inspect`. Khi hiểu shape và đường đi ảnh, đọc `run_epoch` và thực hành smoke. Sau đó mới chốt thí nghiệm cải tiến. Giai đoạn 3 đã bắt đầu bằng nền tảng có thể chạy; chưa tuyên bố hoàn thành toàn bộ góp ý giảng viên.

## Cập nhật quy định chạy 31/10/2026

Từ yêu cầu mới của người dùng: chỉ train trên Google Colab GPU, không train/smoke/unit test có optimizer trên máy cá nhân. Main có chặn nếu thiếu Colab hoặc CUDA. Notebook 02_Colab_GPU_Train.ipynb phục vụ clone/train/lưu kết quả. Kiểm tra local lần cập nhật này chỉ compile cú pháp, không thực thi train. Trạng thái push và cloud run được cập nhật sau khi xác minh.
