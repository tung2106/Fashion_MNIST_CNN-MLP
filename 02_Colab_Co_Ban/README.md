# Giai đoạn 02 — Thực hành Colab cơ bản

**Dành cho người mới học Python:** mở [notebook học chậm mới trên Colab](https://colab.research.google.com/drive/1a0PONFuaDTd-3QD9saWxFUYarCQinwyo), gồm 33 bước nhỏ có giải thích từng dòng. Xem [hướng dẫn bản học chậm](Hoc_Cham/README.md). Bản mới độc lập, không thay notebook và kết quả đã chạy bên dưới.

**Đã chạy:** MLP và CNN, 5 epoch/mô hình, một seed. Chưa có chuỗi thí nghiệm cải tiến của đồ án đầy đủ.

| Tệp/thư mục | Vai trò |
|---|---|
| `Fashion_MNIST_Hoc_Tung_Buoc.ipynb` | Notebook nguồn có giải thích tiếng Việt; bản cục bộ chưa chứa output |
| `ma_nguon_cac_o.py` | Mã các ô để đọc trong VS Code; có tiện ích dành cho Colab |
| `build_notebook.py` | Sinh lại notebook; chạy sẽ ghi đè notebook nguồn cục bộ |
| `ket_qua/20261029_092231_60aa1b` | Lần chạy thật: cấu hình, split, checkpoint, chỉ số, dự đoán và biểu đồ |
| `20261029_092231_60aa1b.zip` | Bản đóng gói gốc tải từ Colab, giữ để dự phòng |
| `ket_qua_colab.jpg` | Ảnh chụp kết quả thật |

Trong mỗi run: `checkpoints` chứa trọng số đã học; `metrics` chứa lịch sử học, dự đoán và chỉ số; `figures` chứa hình. Giữ nguyên chúng cùng một run để truy nguyên số liệu.

Đọc [hướng dẫn từng bước](../TAI_LIEU/03_Huong_Dan_Colab/HUONG_DAN_COLAB.md) và [trạng thái đã kiểm chứng](../TAI_LIEU/03_Huong_Dan_Colab/TRANG_THAI_THUC_HIEN.md). Các thí nghiệm cải tiến tiếp theo chuyển sang giai đoạn 03; không ghi đè kết quả học cơ bản. [Về bản đồ dự án](../README.md).
