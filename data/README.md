# Dữ liệu

- `raw/FashionMNIST/`: ảnh và nhãn gốc, không sửa; không đưa vào Git.
- `dataset.py`: đọc Fashion-MNIST, chia 54k train / 6k validation, tính mean/std từ train, tạo DataLoader. Test có 10k ảnh riêng; tổng 70k.
- Augmentation chỉ cho train. Validation/test/inference dùng cùng thống kê train.

Đường dẫn chung: `utils.paths.DATA_DIR`. Xem ảnh bằng `python -m scripts.inspect_data` (không train; đóng cửa sổ ảnh để lệnh kết thúc). Các split và thống kê thực tế lưu trong từng run tại `experiments/runs/`.
