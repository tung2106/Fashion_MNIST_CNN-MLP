# Trạng thái thực hành Colab

Ngày 29/10/2026. Sau sắp xếp, các tệp mã/kết quả được nhắc dưới đây nằm trong `02_Colab_Co_Ban` tính từ gốc dự án; tài liệu trạng thái này nằm trong `TAI_LIEU/03_Huong_Dan_Colab`.

Notebook đã tạo trong Google Colab: https://colab.research.google.com/drive/1himC9AFXIiUB5KNYW5PlEPnTs7OaOtvL

- Đã tạo và tải lên notebook 28 ô có giải thích tiếng Việt.
- Đã kiểm tra cú pháp các ô Python ở máy cục bộ (không có nbformat nên không chạy validator nbformat).
- Đã Run all và hoàn thành train/test trên Tesla T4 trong Google Colab.
- Run ID: `20261029_092231_60aa1b`; 5 epoch/model, seed 42, batch 128, Adam lr=0.001; train/validation/test = 54k/6k/10k.
- Cả hai checkpoint tốt nhất theo validation loss ở epoch 5. Không chọn theo test.
- MLP: accuracy 0.8750, macro-F1 0.8736778761123951; 101.770 tham số; train khoảng 63,01 giây.
- CNN: accuracy 0.9027, macro-F1 0.9012860677539729; 206.922 tham số; train khoảng 69,28 giây.
- Đã quan sát `TRAIN_COMPLETE`, `EVALUATION_COMPLETE`, `RUN_COMPLETE` và trạng thái notebook đã lưu trên Colab.
- Đã tạo đường cong học, ma trận nhầm lẫn, ảnh sai, bản đồ đặc trưng và demo một ảnh.
- Đã lưu [ZIP kết quả](../../02_Colab_Co_Ban/20261029_092231_60aa1b.zip) về máy bằng cách đọc liên kết xuất tệp hiển thị trong notebook. Tải thông thường qua trình duyệt nhúng không trả về tệp cho công cụ nên dùng ô xuất liên kết ZIP trực tiếp.
- SHA256 ZIP: `67a6a573dde1f1350310c66095683767850c949be1ffcc276730fbe3276e8dec` (khớp Colab).
- Đã giải nén vào `ket_qua/20261029_092231_60aa1b/`; tính lại accuracy/macro-F1 bằng CSV predictions, kiểm tra 10k sample ID duy nhất và hash checkpoint: PASS cho cả hai model.
- `ket_qua_colab.jpg` là ảnh chụp bảng kết quả thật trên Colab.
- Notebook cục bộ là bản nguồn để chạy lại, chưa chứa outputs. Bản trên Colab có outputs thực; chưa tải được bản notebook có outputs qua menu trình duyệt. Checkpoint/predictions/biểu đồ đã có bản sao cục bộ độc lập với runtime.

Hai file đặc tả trong `TAI_LIEU/01_Dac_Ta_Va_Ke_Hoach` vẫn dành cho đồ án đầy đủ; buổi này chỉ MLP/CNN cơ bản, 5 epoch, 1 seed. Không khẳng định đã hoàn thành ablation hoặc toàn bộ đồ án hai tuần. Không dùng kết quả test này để tuning rồi trình bày như test chưa từng được xem.

Đọc kết quả: CNN hơn MLP 2,77 điểm phần trăm accuracy trong lần chạy này. CNN có nhiều tham số hơn, nên không diễn giải đây là bằng chứng kiểm soát hoàn toàn về ưu thế kiến trúc. CNN còn nhầm Shirt → T-shirt/top 157 ảnh, Shirt → Coat 86 ảnh, Shirt → Pullover 73 ảnh; đây là đầu vào tốt để học cách phân tích lỗi.
