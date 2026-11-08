# Cấu trúc dự án thống nhất

Từ 05/11/2026, thư mục được chia theo chức năng. Tên không dấu, chữ thường; package dùng import tuyệt đối từ gốc. Chạy lệnh dạng `python -m package.module` tại gốc dự án.

Ngày 07/11, file code, notebook, cấu hình và tài liệu học tập được đặt tên tiếng Việt không dấu, chữ thường, phân cách bằng dấu gạch dưới. Tên mô tả công việc của file. Số đầu tên notebook thể hiện thứ tự học. Các tên chuẩn như `README.md`, `__init__.py`, `requirements.txt` và tiền tố `test_` được dùng theo quy ước của công cụ.

```text
Phân loại trang phục/
├── models/       Kiến trúc, inference, checkpoints, registry
├── data/         chuan_bi_du_lieu.py và raw/FashionMNIST
├── training/     huan_luyen.py, vong_lap_huan_luyen.py, chon_mo_hinh.py, configs/
├── evaluation/   Chỉ số, biểu đồ, phân tích đặc trưng/lỗi
├── demo/         Mã dự đoán, chay_demo.ps1, kịch bản và examples/; demo offline
├── reports/      Báo cáo Word, thông tin nhóm, figures/, results/ và verification/
├── notebooks/    Bài học, notebook Colab và notebook demo
├── docs/         Hướng dẫn, kiến thức nền, tham khảo
├── experiments/  runs/ bất biến, learning/, gói mã
├── scripts/      Xem dữ liệu, đóng gói mã
├── tests/        Kiểm tra tái cấu trúc; kiểm tra train chỉ Colab GPU
├── utils/        duong_dan.py, kiem_tra_gpu.py, ho_tro_thuc_nghiem.py
├── archive/      Mã/tài liệu cũ chỉ giữ local
├── requirements.txt
└── README.md
```

## Đọc và sửa ở đâu?

| File | Nội dung |
|---|---|
| `data/chuan_bi_du_lieu.py` | Chuẩn bị ảnh, chia tập, chuẩn hóa và gom batch |
| `models/kien_truc_mo_hinh.py` | Kiến trúc MLP, CNN và các biến thể |
| `training/vong_lap_huan_luyen.py` | Vòng học, tính loss, cập nhật trọng số và lưu checkpoint |
| `training/huan_luyen.py` | Chạy huấn luyện theo một cấu hình |
| `training/chon_mo_hinh.py` | So sánh cấu hình và chọn mô hình |
| `evaluation/phan_tich_ket_qua.py` | Phân tích kết quả, ảnh sai và đặc trưng |
| `demo/du_doan_anh.py` | In top 3 dự đoán của một ảnh |
| `demo/so_sanh_mo_hinh.py` | Tạo hình so sánh ba mô hình trên cùng ảnh |
| `demo/chay_demo.ps1` | Gọi dự đoán và mở hình để trình chiếu trên Windows |
| `demo/kich_ban_trinh_bay.md` | Trình tự demo 7 phút, câu hỏi bảo vệ và chuẩn bị máy |
| `scripts/xem_du_lieu.py` | Xem ảnh, nhãn và pixel |
| `scripts/dong_goi_colab.py` | Đóng gói mã nguồn để chạy Colab |
| `utils/kiem_tra_gpu.py` | Kiểm tra môi trường Colab GPU trước khi train |
| `training/configs/moi_truong.json` | Phiên bản thư viện đã kiểm tra |
| `models/du_doan.py` | Nạp checkpoint, chuẩn hóa ảnh và lấy dự đoán |
| `models/danh_sach_mo_hinh.json` | Nguồn và mã kiểm chứng của ba checkpoint |
| `evaluation/danh_gia_mo_hinh.py` | Tính chỉ số trên tập test và lưu dự đoán |
| `evaluation/ve_bieu_do.py` | Vẽ ảnh, đường học, ma trận và đặc trưng |
| `utils/duong_dan.py` | Đường dẫn chung của dự án |
| `utils/ho_tro_thuc_nghiem.py` | Lưu JSON, tính mã kiểm chứng tệp và cố định seed |
| `tests/kiem_tra_du_an.py` | Kiểm tra dữ liệu, cấu trúc và dự đoán trên CPU |
| `tests/kiem_tra_mo_hinh.py` | Đối chiếu artifact và chạy lại 10.000 ảnh test cho mỗi checkpoint demo |
| `tests/kiem_tra_huan_luyen_gpu.py` | Kiểm tra vòng học trên Colab GPU |
| `tests/test_quy_trinh_gpu.py` | Các kiểm tra tự động cho quy trình huấn luyện |

`data.chuan_bi_du_lieu → models.kien_truc_mo_hinh → training.vong_lap_huan_luyen → evaluation.danh_gia_mo_hinh`; `training.huan_luyen`/`training.chon_mo_hinh` điều phối. Demo dùng `models.du_doan`, không import hoặc chạy train. Báo cáo lấy số liệu cùng run/checkpoint, không tự nhập số thay thế.

Các đường dẫn dùng `utils.duong_dan`: ROOT, DATA_DIR, RUNS_DIR, DEFAULT_CHECKPOINT, latest_run(). Không hard-code ổ D hoặc đường dẫn /content trong module; chỉ notebook Colab cấu hình nơi đặt PROJECT. Checkpoint là state_dict nên đổi package không cần train lại.

## Quy tắc giữ kết quả

Run/ZIP/source snapshot cũ giữ nguyên byte. `latest_selection.txt` là con trỏ của máy, được đổi sang tên tương đối. Models/checkpoints là bản sao có kiểm chứng hash của checkpoint đã chọn; registry ghi nguồn. Demo/examples chứa bản sao 7 ảnh thử.

Đã sao lưu mã/tài liệu trước khi chuyển tại `C:/Users/user/Documents/Codex/fashion_restructure_20261105/source_before.zip`; Git vẫn có lịch sử trước đổi. `archive/` giữ các tài liệu/mã bài học cũ không còn dùng làm chương trình hiện hành.

## Skill và nguồn tham khảo

Áp dụng skill scikit-learn (tách dữ liệu/đánh giá, tránh leakage) và jupyter-notebook (giữ ô có mục đích rõ, kiểm tra cú pháp). Các skill được lưu ngoài repo ở `C:/Users/user/.codex/skills/`, không đưa vào Git. Đã tìm hiểu skill pytorch-lightning; dự án hiện dùng PyTorch thuần nên không cần đổi framework để sắp xếp thư mục.

Tham khảo [Cookiecutter Data Science của DrivenData](https://cookiecutter-data-science.drivendata.org/) về phân tách data/models/notebooks/reports và mã dùng lại. Đây là cấu trúc giản lược theo nhu cầu đồ án, không cài thêm template hoặc framework.

## Kiểm tra sau chuyển

`python -m tests.kiem_tra_du_an` chỉ đọc file, kiểm tra split/manifest, import, inference và guard train. Không cập nhật trọng số. `tests/kiem_tra_huan_luyen_gpu.py` và `tests/test_quy_trinh_gpu.py` chứa optimizer nên chỉ chạy Colab GPU. Notebook train mới chưa chạy lại; không gán kết quả cũ cho lần chạy chưa có.

Bằng chứng đợt đổi thư mục: `experiments/reorganization_20261105.json`. Kết quả: hash 253 tệp gốc và 3 bản sao checkpoint khớp; 18 dự đoán khớp; import/cú pháp/liên kết/Ruff PASS. `tests.kiem_tra_du_an` cần data và checkpoint local; khi clone GitHub cần bổ sung chúng trước khi kiểm tra inference.

Kiểm chứng ngày 08/11/2026: `reports/verification/20261108/` lưu biên bản và JSON đối chiếu 30.000 dự đoán test trên ba mô hình. Báo cáo đầy đủ của nhóm nằm ở `reports/bao_cao_phan_loai_thoi_trang_hoan_chinh.docx`, có bản Markdown cùng tên; bản trước vẫn được giữ để đối chiếu.
