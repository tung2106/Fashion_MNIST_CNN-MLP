# Thực hành Fashion-MNIST trên Google Colab

Notebook chính: [Fashion_MNIST_Hoc_Tung_Buoc.ipynb](../../02_Colab_Co_Ban/Fashion_MNIST_Hoc_Tung_Buoc.ipynb).

Đây là buổi học đầu tiên: MLP và CNN tự xây, 5 epoch mỗi mô hình, train 54.000 ảnh, validation 6.000 ảnh, test 10.000 ảnh. Chưa có ablation Dropout/BatchNorm/augmentation hoặc nhiều seed của đặc tả hai tuần. Không coi kết quả cơ bản là đồ án hoàn chỉnh.

## Tệp trong giai đoạn 02

Các tệp mã/kết quả dưới đây nằm trong `02_Colab_Co_Ban` tính từ gốc dự án. Riêng hướng dẫn này và [trạng thái thực hiện](TRANG_THAI_THUC_HIEN.md) nằm trong `TAI_LIEU/03_Huong_Dan_Colab`.

- `Fashion_MNIST_Hoc_Tung_Buoc.ipynb`: 28 ô giải thích/mã, mở bằng Colab.
- `ma_nguon_cac_o.py`: bản mã tương ứng để đọc trong VS Code; có đường dẫn và tiện ích dành cho Colab, không chạy nguyên xi trên Windows.
- `build_notebook.py`: công cụ sinh notebook, không train; chạy lại sẽ ghi đè bản notebook cục bộ, không tác động bản đã lưu trên Colab.
- `TRANG_THAI_THUC_HIEN.md`: trạng thái chạy thực tế và liên kết Colab, được cập nhật sau kiểm chứng.
- `20261029_092231_60aa1b.zip`: gói kết quả thực đã tải từ Colab; bản giải nén nằm trong `ket_qua/20261029_092231_60aa1b/`.
- `ket_qua_colab.jpg`: ảnh bảng kết quả thật. Bản notebook trên Colab có output tại https://colab.research.google.com/drive/1himC9AFXIiUB5KNYW5PlEPnTs7OaOtvL ; bản `.ipynb` cục bộ là nguồn chưa có output.

## Cách chạy

1. Colab → File → Upload notebook → chọn notebook chính.
2. Runtime → Change runtime type → T4 GPU nếu khả dụng. Không mua gói trả phí chỉ để chạy bài này. CPU vẫn dùng được.
3. Đọc từ đầu và chạy từng ô, hoặc Run all để thực hiện trọn buổi học. Đợi ô train hoàn thành trước khi chạy ô phụ thuộc.
4. Kiểm tra `TRAIN_COMPLETE`, `EVALUATION_COMPLETE`, `RUN_COMPLETE`; nếu có lỗi đỏ thì xử lý lỗi đó, không coi ô chưa chạy là có kết quả.
5. Cuối notebook tải ZIP về máy. Notebook lưu Drive và checkpoint trong `/content` là hai loại lưu trữ khác nhau; runtime có thể bị thu hồi.
6. Tải thêm notebook có output bằng File → Download → Download .ipynb nếu muốn xem kết quả ngoại tuyến.

## Công việc và quy trình

| Bước | Nội dung | Bạn cần hiểu | Đầu ra |
|---|---|---|---|
| 1 | Kiểm tra môi trường, cấu hình, thư mục | GPU/CPU, seed, batch, epoch | config.json |
| 2 | Tải và chia dữ liệu | 70k tổng; train/validation/test có vai trò khác nhau | splits.npz, preprocessing.json |
| 3 | Quan sát ảnh/batch | `[B,1,28,28]`, nhãn `[B]`, normalize | ảnh mẫu, shape |
| 4 | Định nghĩa MLP/CNN | Flatten, Conv, ReLU, Pool, Linear, logits | shape mỗi lớp, số tham số |
| 5 | Minh họa một bước học | zero_grad → forward → loss → backward → step | gradient và thay đổi trọng số |
| 6–7 | Train/validation dùng chung | validation không cập nhật trọng số | history.csv, checkpoint best/last |
| 8 | Vẽ đường cong, chốt model | checkpoint có validation loss thấp nhất | curves, frozen_protocol.json |
| 9–10 | Test, bảng/hình và ảnh sai | accuracy/F1/ma trận; kiểm tra cùng nguồn | predictions, metrics, figures |
| 11–12 | Feature maps và dự đoán một ảnh | phản ứng bộ lọc khác giải thích nhân quả | ảnh đặc trưng và demo |
| 13–14 | Tải kết quả, tự kiểm tra | runtime tạm; phạm vi học tập có giới hạn | ZIP, câu hỏi ôn tập |

## Thư mục trên máy Colab

```text
/content/fashion_mnist_basic/
  data/                       ảnh và nhãn gốc; có thể tải lại
  runs/<run_id>/
    config.json               cấu hình và môi trường
    splits.npz                indices train/validation
    preprocessing.json        mean/std từ 54k train
    frozen_protocol.json      dấu vân tay model và cách chọn trước test
    STATUS.json               trạng thái run, giới hạn
    checkpoints/              trọng số tốt nhất và trạng thái cuối
    metrics/                  history, predictions, report, summary
    figures/                  ảnh mẫu, curves, ma trận, ảnh sai, feature maps
  <run_id>.zip                 tải về trước khi kết thúc phiên
```

`data` là đầu vào, `checkpoints` là model đã học, `predictions` là dự đoán từng ảnh, `metrics` là chỉ số tổng hợp và `figures` là hình từ chính kết quả đó. Chạy lại ô cấu hình tạo run mới. Chạy lại ô train trong cùng run sẽ train lại và ghi lại checkpoint của run đó.

## Các khác biệt có chủ đích với đặc tả đầy đủ

- 5 epoch, 1 seed, không tuning, không augmentation/Dropout/BN: học cơ chế trước.
- Chọn best checkpoint bằng validation loss; đồ án đầy đủ dùng quy tắc macro-F1 đã ghi trong file đặc tả. Cần ghi protocol riêng khi so sánh.
- Notebook cơ bản chưa tự resume chính xác toàn bộ RNG giữa các phiên; checkpoint cuối lưu optimizer/history để mở rộng sau.
- Test được dùng sau khi chốt model. Không lấy test làm mục tiêu chỉnh tham số rồi báo là test chưa từng được xem.
- Kết quả chỉ có giá trị sau khi chạy thật; không có mục tiêu accuracy được bảo đảm.

Sau buổi này mới chuyển sang E4/E5/E6 và phân tích sâu theo hai file trong [đặc tả và kế hoạch](../01_Dac_Ta_Va_Ke_Hoach/01_DAC_TA_VA_PROMPT.md). Mã/kết quả cải tiến đặt tại `03_Thuc_Nghiem_Cai_Tien` tính từ gốc dự án. Giữ kết quả buổi học trong run riêng để không trộn với đồ án chính.
