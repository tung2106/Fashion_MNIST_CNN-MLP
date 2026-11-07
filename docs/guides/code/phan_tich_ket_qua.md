# Giải thích kết quả train giai đoạn 3

Nguồn: artifacts/20261031_100341_ac198a38, tải từ Colab ngày 31/10/2026. GPU T4, 54.000 ảnh train, 6.000 validation, batch 128, Adam lr 0.001, 5 epoch/model. Chưa đánh giá test. Không train lại trong lúc tải hoặc phân tích.

## Đọc các cột

- epoch: lượt học qua toàn bộ 54.000 ảnh; 5 epoch là xem lại tập train 5 lượt, không phải có 270.000 ảnh khác nhau.
- train_loss: mức mất mát trung bình lúc học. Nhỏ hơn thường tốt hơn; 0.24 không có nghĩa sai 24%.
- train_accuracy: tỷ lệ đoán đúng trong lúc train. Trọng số thay đổi liên tục giữa các batch.
- val_loss: mất mát trên validation sau epoch, dùng một phiên bản trọng số cố định và không cập nhật chúng.
- val_accuracy: tỷ lệ đoán đúng trên 6.000 ảnh validation không dùng để cập nhật trọng số.

## Kết quả checkpoint được chọn

| Chỉ số | MLP | CNN |
|---|---:|---:|
| Best epoch theo val_loss | 5 | 5 |
| Train loss epoch 5 | 0.286770 | 0.240398 |
| Validation loss epoch 5 | 0.310519 | 0.234121 |
| Train accuracy epoch 5 | 89.45% | 91.29% |
| Validation accuracy epoch 5 | 88.58% | 91.32% |
| Số đúng trên 6.000 validation | 5.315 | 5.479 |
| Số sai | 685 | 521 |
| Tham số học được | 101.770 | 206.922 |
| Thời gian vòng train + validation | 64.16 giây | 71.11 giây |

CNN đoán đúng thêm 164 ảnh validation, hơn 2.73 điểm phần trăm ở checkpoint được chọn. Đây là một lần chạy với hai kiến trúc cụ thể, chưa chứng minh CNN luôn tốt hơn MLP. CNN cũng nhiều tham số hơn nên chưa kiểm soát mọi yếu tố.

## Vì sao chọn epoch 5 của MLP dù accuracy epoch 3 cao hơn?

Epoch 3: accuracy 88.83%, loss 0.314986. Epoch 5: accuracy 88.58%, loss 0.310519. Quy tắc đã chốt là chọn validation loss thấp nhất nên chọn epoch 5. Accuracy chỉ đếm đúng/sai; Cross-Entropy còn xét xác suất dành cho nhãn thật. Hai thước đo không buộc tăng/giảm cùng nhau. Không đổi quy tắc sau khi thấy kết quả chỉ để chọn số đẹp.

## Đọc xu hướng

MLP train loss giảm 0.5022 → 0.2868, validation loss giảm 0.3715 → 0.3105. CNN train loss giảm 0.5210 → 0.2404, validation loss giảm 0.3536 → 0.2341. Điều này cho thấy cả hai đang học được thông tin hữu ích trong lần chạy này. Dao động một epoch không đủ kết luận lỗi chương trình hoặc overfitting.

CNN validation accuracy 91.32% hơi cao hơn training accuracy 91.29% không phải lỗi: training accuracy tích lũy trong lúc trọng số còn thay đổi; validation được đo sau epoch bằng trọng số cố định, và hai tập có ảnh khác nhau. Không so chúng như cùng một phép đo trên cùng mô hình cố định.

Chưa có đủ bằng chứng khẳng định không overfit. Cần nhìn xu hướng nhiều epoch, so sánh train-clean và validation trên cùng checkpoint, và dùng test ở bước đánh giá cuối. Tập test đã từng được xem ở giai đoạn 2; không gọi là holdout hoàn toàn chưa biết.

## Các file dùng để làm gì?

- checkpoints/MLP_best.pt và CNN_best.pt: trọng số chọn theo validation, dùng để dự đoán/đánh giá sau này.
- *_last.pt: trạng thái epoch cuối; bản hiện tại chưa hỗ trợ resume chính xác.
- metrics/*_history.csv: lịch sử từng epoch, nguồn của biểu đồ.
- metrics/*_training.json: best epoch/loss, thời gian, số tham số.
- figures/*_learning_curve.png: đường loss/accuracy theo epoch; train_samples.png là ảnh mẫu.
- config.json: cấu hình thật và thiết bị cuda.
- preprocessing.json: mean/std tính từ train và tên 10 lớp.
- splits.npz: ID ảnh để giữ đúng cách chia.
- source_commit.txt: phiên bản code đã chạy.
- STATUS.json: completed, test_evaluated=false.
- DOWNLOAD_VERIFIED.json: xác nhận ZIP tải khớp SHA256 Colab và kiểm tra CRC thành công (file thêm sau giải nén, không có trong ZIP gốc).

Bước tiếp theo là học cách đọc các bảng này rồi chốt thí nghiệm cải tiến trên validation; không cần train lại để xem kết quả đã có. Mọi lần train tiếp theo chỉ chạy Colab GPU.
