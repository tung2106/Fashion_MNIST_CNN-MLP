# Đánh giá và phân tích

- `danh_gia_mo_hinh.py`: nạp checkpoint đã chốt, lưu predictions, accuracy/macro-F1, report và confusion matrix.
- `ve_bieu_do.py`: ảnh, đường học, ma trận, feature maps.
- `phan_tich_ket_qua.py`: train-clean/validation, cặp nhầm lớp, ảnh sai và kích hoạt tầng đầu/cuối, kiểm tra hash và tính nhất quán.

Lệnh phân tích (không train) từ gốc dự án:

```bash
python -m evaluation.phan_tich_ket_qua --run experiments/runs/<run_id>
```

Lệnh có thể tính lại nhiều dự đoán nên ưu tiên Colab. Kết quả mới nằm trong `analysis/` của run và ZIP được đóng lại; không chạy lại chỉ để xem hình đã có. Báo cáo đã chọn nằm tại [reports/results/20261104](../reports/results/20261104/bao_cao_danh_gia.md).

Feature map chỉ mô tả kích hoạt, không chứng minh một kênh hiểu cổ áo. Softmax chưa được hiệu chuẩn. Test đã được xem từ thực hành ban đầu, không gọi là holdout hoàn toàn chưa biết.
