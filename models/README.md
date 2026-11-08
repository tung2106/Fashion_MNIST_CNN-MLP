# Mô hình

| Tệp/thư mục | Vai trò |
|---|---|
| `kien_truc_mo_hinh.py` | Định nghĩa MLP, CNN, CNNDropout, CNNBatchNorm, CNNPlus |
| `du_doan.py` | Nạp checkpoint, bật eval, tiền xử lý và dự đoán |
| `checkpoints/` | Trọng số đã chọn để demo: `mlp.pt`, `cnn.pt`, `cnn_plus.pt`; JSON chuẩn hóa đi kèm |
| `danh_sach_mo_hinh.json` | Chỉ tới trial/run gốc của từng mô hình |

Bản demo chính là CNNPlus + augmentation, seed 42, 468.010 tham số, checkpoint epoch 30; test accuracy 93,70%. Các bản sao checkpoint có cùng hash với run gốc. Trọng số không được đưa vào Git mặc định; kiến trúc và metadata có trong mã dự án.

Ảnh đầu vào hiện là xám 28×28. `predict_image` trả điểm softmax cho 10 lớp; điểm cao không bảo đảm đúng và chưa được hiệu chuẩn.
