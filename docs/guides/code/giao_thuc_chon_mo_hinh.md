# Chọn mô hình cho demo

Mục tiêu là chọn mô hình tốt trong ngân sách thí nghiệm đã định, kiểm tra độ ổn định và chuẩn bị checkpoint cho giai đoạn 4. Không có bảo đảm đạt độ chính xác cao nhất trong mọi kiến trúc.

## Vì sao cần chạy tiếp?

Mốc cũ 5 epoch đạt 88,58% validation với MLP và 91,32% với CNN. CNN còn cải thiện ở epoch cuối; chưa có test ở giai đoạn 3, chưa tách tác động các kỹ thuật và chưa kiểm tra nhiều seed. Mốc này đủ làm demo thử, chưa đủ kết luận đã tối ưu.

## Thí nghiệm được chốt trước khi xem test

| Thử nghiệm | Thay đổi | Câu hỏi |
|---|---|---|
| mlp | MLP gốc | Đối chứng ảnh trải phẳng |
| cnn | CNN gốc | Lợi ích của kiến trúc tích chập trong thiết lập này |
| cnn_dropout | CNN gốc + Dropout 0,3 | Dropout có giúp không? |
| cnn_batchnorm | CNN gốc + BatchNorm | BatchNorm có giúp không? |
| cnn_aug | CNN gốc + crop/lật nhẹ | Augmentation có giúp không? |
| cnn_plus | CNN 4 lớp tích chập + BN + Dropout | Cấu hình tổng hợp có đáng tăng số tham số? |
| cnn_plus_aug | CNNPlus + augmentation | Augmentation còn giúp kiến trúc này không? |

Tất cả dùng 54k train, 6k validation, split seed 42, Adam 0,001, batch 128, tối đa 30 epoch. Mean/std chỉ tính từ train. Augmentation chỉ tác động train. Các thử nghiệm ban đầu dùng seed khởi tạo 42. Cùng seed không bảo đảm trùng từng bit giữa mọi thiết bị/phiên bản.

- **Checkpoint:** chọn validation loss nhỏ nhất, hòa giữ epoch sớm hơn.
- **Dừng sớm:** dừng sau 6 epoch không cải thiện loss ít nhất 0,0001 so với mốc cải thiện gần nhất. Checkpoint vẫn lưu mỗi lần loss lập mức thấp mới.
- **Giảm learning rate:** ReduceLROnPlateau giảm một nửa sau khi hết patience 2; tối thiểu 0,00001. LR giảm trước khi điều kiện dừng sớm thường được kích hoạt.
- **Chọn cấu hình:** macro-F1 validation cao nhất ở checkpoint đã chọn; hòa thì loss thấp hơn.
- **Độ ổn định:** chạy cấu hình thắng với thêm seed 123 và 2026, giữ nguyên split. Báo trung bình ± độ lệch chuẩn mẫu của 3 seed; đây không phải khoảng tin cậy hay bằng chứng về mọi lần chạy.
- **Demo:** dùng seed 42 đã định trước; không chọn seed có test đẹp nhất.
- **Test:** chốt hash checkpoint rồi mới đánh giá 10k ảnh. Test đã được xem ở giai đoạn 2 nên không gọi là holdout hoàn toàn chưa biết. Không chỉnh mô hình dựa trên test.

## Đáp ứng góp ý và bài tham khảo

Ảnh nhận xét của cô chỉ đọc được phần thiếu phân tích feature engineering và lời khen về dữ liệu, CNN, MLP, Adam/Cross-… Phần sau bị cắt, không suy diễn thành rubric đầy đủ.

Nhóm trước đã có learning curves, confusion matrix và ảnh sai. Điểm cần làm sâu hơn là phân biệt **tiền xử lý** (đổi giá trị/biến đổi ảnh) với **đặc trưng được học** (các bản đồ kích hoạt), giải thích những lớp dễ nhầm và kiểm chứng hiệu quả bằng đối chứng. Đề tài mới dùng ảnh trung bình từng lớp, feature maps nhiều tầng trên ảnh đúng/sai, cùng bảng nhầm có hướng. Không gán chắc chắn một kênh là bộ lọc cổ áo nếu chưa kiểm chứng.

Báo cáo cũ có số liệu giữa lời, bảng và hình chưa khớp. Mã mới lưu dự đoán từng ảnh; tính tất cả bảng và hình từ cùng checkpoint. Kiểm tra tổng ma trận, đường chéo/10.000, support từng lớp và hash trọng số. Không sao chép số liệu của nhóm trước làm kết quả của nhóm mình.

## Đọc mã theo thứ tự

1. `data/chuan_bi_du_lieu.py`: ảnh → split → chuẩn hóa → batch.
2. `models/kien_truc_mo_hinh.py`: ảnh → các lớp mạng → 10 logits.
3. `training/vong_lap_huan_luyen.py`: logits + nhãn → Cross-Entropy → gradient → Adam cập nhật trọng số; validation không cập nhật.
4. `training/chon_mo_hinh.py`: điều phối các thử nghiệm theo quy tắc trên.
5. `evaluation/danh_gia_mo_hinh.py` và `evaluation/phan_tich_ket_qua.py`: đo chất lượng, kiểm chứng bảng/hình, phân tích lỗi.
6. `models/du_doan.py`: chỉ nạp trọng số và dự đoán; dùng lại cho demo.

Adam dùng thông tin gradient hiện tại và tích lũy để điều chỉnh bước cập nhật từng tham số. Cross-Entropy phạt việc dành xác suất thấp cho nhãn đúng; nó nhận logits và nhãn, không so sánh từng pixel. Softmax dùng hiển thị điểm dự đoán, không đặt thêm trước CrossEntropyLoss.

## Phạm vi đánh giá

Đối chứng Dropout/BN/augmentation ban đầu chỉ có một seed, vì vậy không khẳng định hiệu ứng nhỏ chắc chắn có ý nghĩa thống kê. CNNPlus thay nhiều yếu tố cùng lúc nên không dùng để tách riêng tác động BN hoặc Dropout. So sánh thời gian/số tham số cùng accuracy để tránh chỉ chạy mô hình lớn hơn.

Demo đầu tiên dùng ảnh xám 28×28 của Fashion-MNIST. Ảnh điện thoại có nền, màu, góc chụp khác là dữ liệu ngoài phân phối; cần kiểm chứng riêng. Điểm softmax cao vẫn có thể đi kèm dự đoán sai.

Nguồn: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html), [ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html). Skill hỗ trợ quy trình: [scikit-learn trong Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills/tree/main/skills/scikit-learn), cài ngoài Git dự án.
