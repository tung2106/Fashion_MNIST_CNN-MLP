# Dữ liệu

- `raw/FashionMNIST/`: ảnh và nhãn gốc, không sửa; không đưa vào Git.
- `chuan_bi_du_lieu.py`: đọc Fashion-MNIST, chia 54k train / 6k validation, tính mean/std từ train, tạo DataLoader. Test có 10k ảnh riêng; tổng 70k.
- Augmentation chỉ cho train. Validation/test/inference dùng cùng thống kê train.

Đường dẫn chung: `utils.duong_dan.DATA_DIR`. Xem ảnh bằng `python -m scripts.xem_du_lieu` (không train; đóng cửa sổ ảnh để lệnh kết thúc). Các split và thống kê thực tế lưu trong từng run tại `experiments/runs/`.
