# Trạng thái thực nghiệm

## Cập nhật 05/11/2026 — cấu trúc dự án thống nhất

Mã chia theo models/data/training/evaluation/demo; báo cáo ở reports, notebook ở notebooks, artifact gốc ở experiments/runs. Xem [cấu trúc](../../project_structure.md). Kết quả ngày 04/11 và dữ liệu gốc giữ nguyên; các con số cũ không đại diện một lần train mới. Notebook Drive giữ nguyên. Bản mã hiện hành gồm đợt tái cấu trúc và demo offline ngày 06/11.

Đã kiểm chứng hash 253 tệp dữ liệu/kết quả bất biến và 3 bản sao checkpoint. 18 phép đối chiếu dự đoán (6 ảnh × MLP/CNN/CNNPlus) khớp predictions gốc trong dung sai; CLI mới chạy thành công. Kiểm tra import, cú pháp notebook, Ruff và liên kết tài liệu PASS. Không thực hiện train local; notebook train theo cấu trúc mới chưa chạy lại trên Colab.

## Cập nhật 04/11/2026 — đã train và đánh giá trên Colab

- Colab T4: https://colab.research.google.com/drive/1K9FNgM9mzUL55AmOaVcc_vIfLd5DKkP0.
- Đã triển khai 7 cấu hình, early stopping, giảm learning rate, chọn cấu hình bằng validation và chạy thêm 2 seed cho cấu hình thắng.
- Run chính `selection_20261104_144952_e58647`: đã hoàn thành 9 lượt train trên Tesla T4. Cấu hình được chọn bằng validation: CNNPlus + augmentation, seed 42, checkpoint epoch 30, 468.010 tham số.
- Accuracy validation 94,33%, macro-F1 validation 94,30%. Accuracy test MLP 88,55%, CNN cơ bản 92,11%, bản demo 93,70%. Ba seed của cấu hình được chọn: 93,70% / 93,84% / 93,87%; trung bình 93,80%, độ lệch chuẩn mẫu 0,091 điểm phần trăm.
- Kiểm tra kỹ thuật trên Colab PASS: split, shape, train/eval, early stopping, scheduler và checkpoint reload. Phân tích đã hoàn tất; kiểm tra số liệu và inference từng ảnh khớp đánh giá PASS.
- Kiểm tra cú pháp/Ruff trên các file mới PASS; không thực hiện train hoặc test có optimizer trên máy cá nhân.
- Skill `scikit-learn` đã cài ngoài repository. `codex.md` ghi yêu cầu giảng viên và hướng phát triển; `AGENTS.md` yêu cầu các phiên sau đọc file đó. Cả hai là file local, được Git bỏ qua.
- Lượt khởi động thử trước đó bị dừng để bổ sung đủ đối chứng; không dùng kết quả dở làm số liệu cuối. Lượt chính có log từng epoch và source snapshot.
- ZIP đã được người dùng tải vào `experiments/runs/selection_20261104_144952_e58647.zip`, 45.321.073 byte. SHA256 `0eeb6ec013f881a77bef3ef6019e9fcdf25ff1c458074c25ca0f195bc264b90c` khớp Colab, CRC PASS; đã giải nén 172 tệp vào thư mục cùng tên. Có `DOWNLOAD_VERIFIED.json`.
- Inference CPU trên 6 ảnh (4 đúng, 2 sai) khớp Colab trong dung sai; CLI chạy thành công. Đã tính lại accuracy/macro-F1, kiểm tra ma trận/support/hash cho cả 5 bảng test. Có `LOCAL_INFERENCE_VERIFIED.json`; không train trên máy cá nhân.
- [Báo cáo chuẩn bị demo](../../../reports/results/20261104/danh_gia_chuan_bi_demo.md) đã bổ sung bảng, hình, phân tích đặc trưng/lỗi và đối chiếu yêu cầu. Đủ checkpoint để bắt đầu demo Fashion-MNIST; chưa có giao diện hoàn chỉnh. Bản cập nhật Git gồm mã, notebook, tài liệu và hình/bảng đã kiểm chứng; ZIP, trọng số và hướng dẫn AI giữ local.
- Bản được chọn chạm ngân sách 30 epoch; không tuyên bố tối ưu tuyệt đối. Test đã được xem từ giai đoạn 2; không dùng test để tiếp tục chọn cấu hình.

Các mục ngày 31/10 bên dưới là lịch sử, không thay thế giao thức mới tại `giao_thuc_chon_mo_hinh.md`.

## Đã thực hiện

- Tách mã cơ bản từ notebook giai đoạn 2 thành data/models/engine/evaluate/plots/utils và main.py.
- Tạo cấu hình baseline, smoke; lệnh inspect không train, train mặc định không xem test.
- Tạo môi trường `.venv`, dùng lại torch/torchvision có sẵn; các gói bổ sung cài vào venv. Phiên bản đã kiểm tra được ghi tại `training/configs/moi_truong_da_kiem_tra.json`.
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
`experiments/runs/20261031_105329_eed4fb16`.
320 ảnh train / 80 validation; mỗi mô hình 1 epoch; CPU. MLP và CNN đều hoàn thành, có best/last checkpoint, history và đường cong. `STATUS.json` ghi completed, profile smoke, test_evaluated false. Không dùng các chỉ số run này làm số liệu chất lượng đồ án.

**Xem dữ liệu đầy đủ (inspect):**
`experiments/runs/20261031_105521_a4175a5a`.
54.000 train / 6.000 validation, batch ảnh [128,1,28,28], nhãn [128]. Có splits, preprocessing, config và ảnh train_samples.png đã kiểm tra trực quan. Không train và không xem test.

Artifacts giữ trên máy và được loại khỏi Git mặc định. Các đường dẫn trên tính từ gốc dự án; nếu clone repository ở máy khác, chạy lại lệnh để tạo artifact mới. Tài liệu này ghi lại bằng chứng đã chạy trên máy hiện tại, không giả định các file local đi kèm repository.

## Chưa thực hiện ở giai đoạn 3

- Đánh giá test 10k của mã module mới; huấn luyện đầy đủ đã hoàn tất theo cập nhật bên dưới.
- Ablation Dropout, BatchNorm, augmentation; nhiều seed; phân tích đặc trưng/lỗi sâu.
- Chức năng resume chính xác, evaluate-only và ứng dụng demo.
- Mã giai đoạn 3 đã push GitHub; xem cập nhật bên dưới.

Giữ tiêu chí min validation loss để khớp notebook học tập. Khi chuyển sang thí nghiệm cải tiến với tiêu chí macro-F1 theo đặc tả, cần ghi giao thức mới và chạy đối chứng cùng tiêu chí. Không dùng test để chọn cấu hình.

## Bước học tiếp theo

Đọc `cach_doc_ma_nguon.md` phần 1–3 rồi chạy `main.py --inspect`. Khi hiểu shape và đường đi ảnh, đọc `run_epoch` và thực hành smoke. Sau đó mới chốt thí nghiệm cải tiến. Giai đoạn 3 đã bắt đầu bằng nền tảng có thể chạy; chưa tuyên bố hoàn thành toàn bộ góp ý giảng viên.

## Cập nhật quy định chạy 31/10/2026

Từ yêu cầu mới của người dùng: chỉ train trên Google Colab GPU, không train/smoke/unit test có optimizer trên máy cá nhân. Main có chặn nếu thiếu Colab hoặc CUDA. Notebook 02_train_gpu.ipynb phục vụ clone/train/lưu kết quả. Kiểm tra local lần cập nhật này chỉ compile cú pháp, không thực thi train. Trạng thái push và cloud run được cập nhật sau khi xác minh.

## Đã train đầy đủ trên Colab GPU — 31/10/2026

- [Notebook đã chạy và lưu trên Drive](https://colab.research.google.com/drive/1WiCTnJhuDdR6ovY-NCgWCHWbYDRqWdjH).
- Commit mã huấn luyện: `1142ebf2f3edebf0599c6613d6bc563a04a08fed`.
- Thiết bị Tesla T4; PyTorch 2.11.0+cu130. Không train trên máy cá nhân trong lần này.
- Run ID trên máy ảo UTC: `20261031_100341_ac198a38` (17:03 giờ Việt Nam).
- 54k train / 6k validation; 5 epoch/model; không đánh giá test.
- MLP: best epoch 5, validation loss 0.31051852472623187, validation accuracy 0.885833 (bảng hiển thị làm tròn), 101770 tham số, train+validation 64.16 giây.
- CNN: best epoch 5, validation loss 0.2341205857594808, validation accuracy 0.913167 (bảng hiển thị làm tròn), 206922 tham số, train+validation 71.11 giây.
- Mốc TRAIN_COMPLETE đã xuất hiện, cả hai history đủ 5 epoch, ô đọc bảng/biểu đồ và đóng ZIP đã chạy.
- Gói kết quả trên Colab: `/content/fashion_stage3_9e26351e/03_Thuc_Nghiem_Cai_Tien/artifacts/20261031_100341_ac198a38.zip`.
- Đã tải ZIP về artifacts/20261031_100341_ac198a38.zip và giải nén vào thư mục cùng tên ngày 31/10/2026. SHA256 khớp bản Colab, CRC toàn bộ ZIP PASS. Có DOWNLOAD_VERIFIED.json; không train lại.
- Bản notebook trên GitHub sau đó bổ sung hiển thị log subprocess theo dòng; thay đổi hiển thị này chưa chạy lại và không thay thuật toán.
