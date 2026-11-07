# Cấu trúc dự án thống nhất

Từ 05/11/2026, thư mục được chia theo chức năng. Tên không dấu, chữ thường; package dùng import tuyệt đối từ gốc. Chạy lệnh dạng `python -m package.module` tại gốc dự án.

```text
Phân loại trang phục/
├── models/       Kiến trúc, inference, checkpoints, registry
├── data/         dataset.py và raw/FashionMNIST
├── training/     train.py, engine.py, select_model.py, configs/
├── evaluation/   Chỉ số, biểu đồ, phân tích đặc trưng/lỗi
├── demo/         predict.py, present.py và examples/; demo offline
├── reports/      Báo cáo, slide và results/ đã chọn
├── notebooks/    Bài học, notebook Colab và notebook demo
├── docs/         Hướng dẫn, kiến thức nền, tham khảo
├── experiments/  runs/ bất biến, learning/, gói mã
├── scripts/      Xem dữ liệu, đóng gói mã
├── tests/        Kiểm tra tái cấu trúc; kiểm tra train chỉ Colab GPU
├── utils/        paths.py, runtime.py, io.py
├── archive/      Mã/tài liệu cũ chỉ giữ local
├── requirements.txt
└── README.md
```

## Đọc và sửa ở đâu?

`data.dataset → models.networks → training.engine → evaluation.evaluate`; `training.train`/`training.select_model` điều phối. Demo dùng `models.inference`, không import hoặc chạy train. Báo cáo lấy số liệu cùng run/checkpoint, không tự nhập số thay thế.

Các đường dẫn dùng `utils.paths`: ROOT, DATA_DIR, RUNS_DIR, DEFAULT_CHECKPOINT, latest_run(). Không hard-code ổ D hoặc đường dẫn /content trong module; chỉ notebook Colab cấu hình nơi đặt PROJECT. Checkpoint là state_dict nên đổi package không cần train lại.

## Quy tắc giữ kết quả

Run/ZIP/source snapshot cũ giữ nguyên byte. `latest_selection.txt` là con trỏ của máy, được đổi sang tên tương đối. Models/checkpoints là bản sao có kiểm chứng hash của checkpoint đã chọn; registry ghi nguồn. Demo/examples chứa bản sao 6 ảnh thử.

Đã sao lưu mã/tài liệu trước khi chuyển tại `C:/Users/user/Documents/Codex/fashion_restructure_20261105/source_before.zip`; Git vẫn có lịch sử trước đổi. `archive/` giữ các tài liệu/mã bài học cũ không còn dùng làm chương trình hiện hành.

## Skill và nguồn tham khảo

Áp dụng skill scikit-learn (tách dữ liệu/đánh giá, tránh leakage) và jupyter-notebook (giữ ô có mục đích rõ, kiểm tra cú pháp). Các skill được lưu ngoài repo ở `C:/Users/user/.codex/skills/`, không đưa vào Git. Đã tìm hiểu skill pytorch-lightning; dự án hiện dùng PyTorch thuần nên không cần đổi framework để sắp xếp thư mục.

Tham khảo [Cookiecutter Data Science của DrivenData](https://cookiecutter-data-science.drivendata.org/) về phân tách data/models/notebooks/reports và mã dùng lại. Đây là cấu trúc giản lược theo nhu cầu đồ án, không cài thêm template hoặc framework.

## Kiểm tra sau chuyển

`python -m tests.verify_layout` chỉ đọc file, kiểm tra split/manifest, import, inference và guard train. Không cập nhật trọng số. `tests/verify_training_gpu.py` và `tests/test_pipeline_gpu.py` chứa optimizer nên chỉ chạy Colab GPU. Notebook train mới chưa chạy lại; không gán kết quả cũ cho lần chạy chưa có.

Bằng chứng đợt đổi thư mục: `experiments/reorganization_20261105.json`. Kết quả: hash 253 tệp gốc và 3 bản sao checkpoint khớp; 18 dự đoán khớp; import/cú pháp/liên kết/Ruff PASS. `verify_layout` cần data và checkpoint local; khi clone GitHub cần bổ sung chúng trước khi kiểm tra inference.
