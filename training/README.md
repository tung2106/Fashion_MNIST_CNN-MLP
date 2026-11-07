# Huấn luyện

- `train.py`: một cấu hình; `--inspect` chỉ xem dữ liệu.
- `engine.py`: forward, loss, backward, Adam; validation, scheduler, early stopping và lưu checkpoint.
- `select_model.py`: so sánh 7 cấu hình, xác nhận cấu hình chọn qua 3 seed, chốt rồi test.
- `configs/`: baseline 5 epoch, selection tối đa 30 epoch, smoke tập nhỏ.

Chạy từ gốc dự án **trên Colab GPU**:

```bash
python -m training.train --config configs/baseline.json
python -m training.select_model
```

`python -m training.train --inspect` chỉ đọc/chia/hiển thị dữ liệu và có thể chạy local. `utils.runtime` chặn các lệnh train và vòng học cập nhật trọng số ngoài Colab GPU.

Đầu vào: `data/raw/`. Đầu ra: `experiments/runs/<run_id>/`. Chọn checkpoint bằng validation loss; chọn cấu hình bằng macro-F1 validation; không chỉnh mô hình bằng test. Xem [giao thức](../docs/guides/code/giao_thuc_chon_mo_hinh.md).
