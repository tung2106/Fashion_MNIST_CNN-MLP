# Dữ liệu dùng chung

`FashionMNIST/raw/` chứa ảnh và nhãn gốc Fashion-MNIST: 60.000 train gốc và 10.000 test. Tách 60.000 thành 54.000 train và 6.000 validation trong mã; không cần di chuyển từng ảnh thành các thư mục mới.

Các tệp `train-*` dành cho train gốc; `t10k-*` dành cho test. `images` là ảnh, `labels` là nhãn. Mỗi loại có bản `.gz` nén và bản giải nén; đây là cấu trúc loader tải về, giữ nguyên cả hai.

`01_Lam_Quen_Du_Lieu/xem_du_lieu.py` đã trỏ tới thư mục này bằng vị trí tệp, không phụ thuộc thư mục terminal hiện tại. Colab có ổ đĩa riêng và tiếp tục dùng `/content/fashion_mnist_basic/data`.

Không lưu model hoặc kết quả dự đoán vào đây. [Về bản đồ dự án](../README.md).
