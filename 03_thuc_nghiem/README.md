# Giai đoạn 3 — Chương trình MLP/CNN có cấu trúc

Đã triển khai phần nền tảng, tách mã notebook thành các module. Chưa hoàn thành toàn bộ thí nghiệm cải tiến. Giữ nguyên notebook, kết quả và phiên Colab giai đoạn 2.

## Cấu trúc

```text
03_thuc_nghiem/
  main.py                   Điểm bắt đầu, điều phối các phần
  requirements.txt          Thư viện cần cài
  configs/
    baseline.json           54k train / 6k validation, 5 epoch, batch 128
    smoke.json              320 train / 80 validation, 1 epoch kiểm tra kỹ thuật
  src/
    __init__.py             Đánh dấu package Python
    data.py                 Đọc ảnh, chia tập, chuẩn hóa, DataLoader
    models.py               Kiến trúc MLP và CNN như notebook cũ
    engine.py               Train, validation, chọn best checkpoint
    evaluate.py             Nạp best, dự đoán test, tính và lưu chỉ số
    plots.py                Vẽ ảnh, đường cong, ma trận nhầm, feature maps
    utils.py                Seed, JSON và mã nhận diện file SHA256
  notebooks/
    01_chay_module.ipynb    Gọi từng bước từ mã chung
    02_train_gpu.ipynb      Clone mã dự án và train trên Colab GPU
  tests/test_pipeline.py    Kiểm tra tính đúng
  artifacts/<run_id>/       Kết quả riêng của từng lần chạy
    config.json             Cấu hình và phiên bản thực tế
    preprocessing.json      Mean/std, nhãn và số ảnh mỗi lớp
    splits.npz              ID ảnh train/validation
    STATUS.json             Trạng thái hoàn thành hoặc lỗi
    checkpoints/            Trọng số best và last
    metrics/                History; predictions/report khi thực sự test
    figures/                Các hình sinh từ run này
```

Dữ liệu gốc dùng chung ở `../du_lieu_chung`. Đọc [giải thích từng phần](../tai_lieu_du_an/huong_dan_thuc_nghiem/cach_doc_ma_nguon.md) và [trạng thái kiểm tra](../tai_lieu_du_an/huong_dan_thuc_nghiem/tien_do_giai_doan_03.md).

## Huấn luyện chỉ trên Colab GPU

Theo yêu cầu của chủ dự án, không train trên máy cá nhân, kể cả smoke. `main.py` từ chối huấn luyện nếu không chạy trong Google Colab hoặc không có CUDA. Các smoke đã ghi trước đây là kiểm tra lịch sử trước yêu cầu này.

Dùng [notebook GPU trong dự án](notebooks/02_train_gpu.ipynb) hoặc [bản trên Drive](https://colab.research.google.com/drive/1trdCuk8Sp50puR1VdrtAkPijWdwKUqcE), lưu bản sao vào Drive, chọn Runtime → Change runtime type → T4 GPU rồi Run all.

Notebook tự clone mã GitHub vào thư mục mới trên máy ảo Google, xác nhận GPU, train MLP/CNN đầy đủ 5 epoch/model và tải ZIP kết quả. Chưa đánh giá test để giữ test ngoài quá trình lựa chọn cấu hình. Ghi commit nguồn trong run để truy vết.

Máy cá nhân chỉ sửa/đọc mã và xem dữ liệu. Lệnh xem dữ liệu từ gốc dự án (không train):

```powershell
.\.venv\Scripts\python.exe .\03_thuc_nghiem\main.py --inspect
```

Các kiểm tra có bước optimizer trong tests cũng chỉ chạy trên Colab theo quy trình hiện tại. Không chạy unittest hoặc smoke trên máy cá nhân. Môi trường local đã tạo trước đây không cần dùng để train.

## Giao thức và giới hạn

- Baseline giữ nguyên seed 42, stratified 54k/6k, Adam 0.001, 5 epoch; mean/std chỉ tính từ train. Test gốc 10k chỉ đọc khi bật cờ đánh giá.
- Chọn best theo **validation loss thấp nhất**, hòa giữ epoch sớm hơn, giống notebook. Đặc tả thí nghiệm cải tiến định hướng macro-F1: khi triển khai E2–E6 cần chốt giao thức mới trước và chạy baseline đối chứng cùng tiêu chí, không đổi âm thầm.
- Smoke chỉ kiểm tra phần mềm trên tập nhỏ; không dùng accuracy smoke làm số liệu đồ án. Không test trong smoke.
- Best dùng đánh giá. Last lưu trạng thái cuối nhưng chưa hỗ trợ resume chính xác. Cùng seed không đảm bảo giống từng bit giữa các phiên bản/thiết bị.
- Test đã được xem ở giai đoạn 2, không gọi là holdout chưa từng biết.
- Feature maps mới minh họa 4 kênh Conv đầu trên test ID 0. Phân tích sâu, ablation Dropout/BN/augmentation, demo và báo cáo còn ở các bước tiếp theo; chưa hoàn tất góp ý của cô.
- Artifacts được loại khỏi Git mặc định; chỉ chọn số liệu/hình cần nộp sau khi kiểm tra. AI skills tiếp tục được loại theo .gitignore.

## Lần train GPU đã hoàn thành

[Notebook kết quả trên Drive](https://colab.research.google.com/drive/1WiCTnJhuDdR6ovY-NCgWCHWbYDRqWdjH): MLP/CNN đã train 5 epoch trên T4. Accuracy validation tại best checkpoint: 88,58% / 91,32%; chưa test. Xem [tiến độ](../tai_lieu_du_an/huong_dan_thuc_nghiem/tien_do_giai_doan_03.md) để tra commit, run và ZIP; xem [phân tích kết quả](../tai_lieu_du_an/huong_dan_thuc_nghiem/phan_tich_ket_qua.md) để đọc các chỉ số.
