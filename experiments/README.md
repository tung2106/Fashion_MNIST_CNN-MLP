# Thực nghiệm gốc

- `runs/<run_id>/`: kết quả baseline và lựa chọn mô hình; mỗi run có config, preprocessing, splits, checkpoint, metrics, figures và source đã chạy.
- `learning/`: kết quả thực hành ban đầu, chỉ giữ để tham khảo.
- `source.zip`: gói mã hiện tại tạo bằng `python -m scripts.pack_source`.

Run được chọn: `runs/selection_20261104_144952_e58647/`. `runs/latest_selection.txt` chứa tên run tương đối để chuyển máy vẫn dùng được.

Khi chuyển thư mục, toàn bộ 253 tệp dữ liệu/kết quả đã đối chiếu hash. Không sửa lại config/source snapshot/log/ZIP lịch sử: chúng mô tả đường dẫn của phiên thực tế đã chạy. Mã hiện hành nằm ở các thư mục models/data/training/evaluation, không chạy source cũ trong artifact.

Các run/ZIP giữ local theo `.gitignore`; bảng/hình cần nộp đã chọn vào `reports/results/20261104/`.
