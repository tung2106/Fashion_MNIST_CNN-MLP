# Hiểu mã giai đoạn 3 từ đầu

**Quy định mới:** mọi thao tác train, smoke hoặc kiểm tra có cập nhật trọng số bên dưới chỉ thực hiện trên Google Colab GPU. Máy cá nhân chỉ đọc/sửa mã và inspect. Main đã chặn train ngoài Colab GPU.

## Tại sao tách file?

Notebook giữ biến giữa các ô. Chương trình Python gọi các hàm có đầu vào/đầu ra rõ ràng, không yêu cầu nhớ ô nào đã chạy. Chia file không tự tăng accuracy; nó giúp sửa đúng nơi, kiểm tra và dùng lại cho demo.

Luồng: **main → data → models → engine → evaluate → plots**. `utils` hỗ trợ nhiều phần. Thứ tự đọc nên là cấu hình → dữ liệu → mô hình → vòng học → đánh giá → main. Chưa cần hiểu argparse hay SHA256 trước khi học vòng train.

## 1. configs: các lựa chọn của thí nghiệm

`baseline.json` là bảng khóa–giá trị: seed 42, batch size 128, epochs 5, lr 0.001. `config['epochs']` đọc giá trị 5. Batch là một nhóm ảnh, epoch là một lượt qua train, learning rate ảnh hưởng quy mô cập nhật, seed điều khiển nguồn ngẫu nhiên để dễ lặp lại.

`smoke.json` dùng 320 train/80 validation, 1 epoch để tìm lỗi chương trình nhanh. Số liệu đó không đại diện chất lượng đồ án.

## 2. data.py: chuẩn bị dữ liệu

```python
train_ids, val_ids = split_indices(raw.targets, config['seed'])
```

`raw.targets` là nhãn thật của 60.000 ảnh train gốc. Dấu chấm đọc thuộc tính. Hàm `split_indices` trả hai mảng ID; dấu phẩy nhận chúng vào hai biến. ID là vị trí ảnh, không phải nhãn: ID 100 có thể mang nhãn 8.

`stratify` giữ tỷ lệ lớp: baseline có 5.400 ảnh/lớp trong train, 600/lớp trong validation. Hai tập không giao nhau. Test là 10.000 ảnh riêng do Fashion-MNIST cung cấp; tổng dữ liệu là 70.000 ảnh.

```python
pixels = raw.data[train_ids].float() / 255.0
```

Chọn pixel chỉ của train, đổi sang số thực, đưa thang 0–255 về 0–1. Mean là trung bình; std là độ lệch chuẩn. Normalize tính `(pixel - mean) / std`, nên đầu ra có thể âm. Đây là tiền xử lý, không phải học trọng số. Không lấy thống kê từ validation/test.

`Subset` chọn ảnh theo ID. `DataLoader` gom ảnh thành batch. `shuffle=True` đổi thứ tự ảnh train; không làm thay đổi nhãn và không chia lại tập. Với 54.000 ảnh, batch 128 có 422 batch, batch cuối có 112 ảnh.

## 3. models.py: cấu trúc mạng

`class MLP(nn.Module)` định nghĩa mạng kế thừa khả năng quản lý tham số của PyTorch. `__init__` tạo các lớp khi gọi `MLP()`. `self` là đối tượng mạng hiện tại; `super().__init__()` khởi tạo chức năng lớp cha. `nn.Sequential` xếp lớp theo thứ tự. `forward(self, x)` mô tả đường đi khi gọi `model(images)`.

| Bước MLP | Shape | Bước CNN | Shape |
|---|---|---|---|
| Ảnh | [B,1,28,28] | Ảnh | [B,1,28,28] |
| Flatten | [B,784] | Conv + ReLU | [B,16,28,28] |
| Linear + ReLU | [B,128] | MaxPool | [B,16,14,14] |
| Linear | [B,10] | Conv + ReLU + Pool | [B,32,7,7] |
| | | Flatten | [B,1568] |
| | | Linear/ReLU/Linear | [B,10] |

B là số ảnh, 128 trong Linear là số nút chứ không phải batch size. ReLU thay số âm bằng 0. Conv dùng chung bộ lọc trên nhiều vùng ảnh; Pool giảm kích thước không gian. CNN cũng Flatten nhưng sau khi học đặc trưng.

Trọng số được PyTorch khởi tạo khi tạo model. Train sửa trọng số; optimizer không sửa pixel đầu vào. MLP ở đây có 101.770 tham số, CNN 206.922: CNN không luôn ít tham số hơn MLP. Cả hai trả **logits**: 10 điểm lớp mỗi ảnh, không phải pixel hoặc phần trăm.

## 4. engine.py: vòng học quan trọng nhất

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

1. Xóa gradient batch trước, giữ nguyên trọng số đã học.
2. Forward: đưa ảnh qua mạng bằng trọng số hiện tại để có 10 điểm mỗi ảnh.
3. CrossEntropyLoss đối chiếu điểm lớp với nhãn thật. Không so logits với pixel; loss không phải tỷ lệ sai. Truyền logits trực tiếp, không Softmax trước hàm này.
4. Tính gradient: loss nhạy thế nào với thay đổi mỗi tham số. Dòng này chưa sửa trọng số.
5. Adam dùng gradient và thông tin tích lũy để cập nhật tham số. Learning rate ảnh hưởng mức cập nhật; không đảm bảo mọi ảnh đều tốt hơn sau mỗi bước.

`run_epoch` lặp qua các batch. `fit` gọi nhiều epoch, đo validation sau mỗi epoch, lưu checkpoint tốt nhất theo validation loss. Khi hòa giữ epoch sớm hơn.

Validation không truyền optimizer. Hàm chọn eval mode, tắt ghi đồ thị đạo hàm, bỏ backward/step. `model.eval()` một mình không tắt gradient. `model.train()` một mình cũng chưa học.

Loss batch là trung bình; batch cuối có thể nhỏ, nên cộng `loss * số ảnh` rồi chia tổng số ảnh. Không lấy trung bình đơn giản các batch khác kích thước.

## 5. evaluate.py: kiểm tra chất lượng

Checkpoint lưu `state_dict` (tên tham số → tensor), model, epoch, mean/std và tên lớp. Nạp trọng số vào đúng kiến trúc mới khôi phục được mạng. Best là tốt nhất theo validation; last là trạng thái cuối, có thể kém best. Bản hiện tại chưa tự resume.

Hàm đánh giá nạp **best**, Softmax đổi logits thành xác suất rồi `argmax` lấy vị trí lớn nhất. Vị trí 8 là Bag. Xác suất cao vẫn có thể sai.

- Accuracy = số ảnh đoán đúng / tổng ảnh.
- Macro-F1 = trung bình F1 của 10 lớp, trọng số ngang nhau.
- Confusion matrix: hàng nhãn thật, cột dự đoán; đường chéo là số đúng.
- Kiểm tra tổng ma trận = 10.000 và tổng đường chéo / 10.000 = accuracy.
- CSV predictions chứa ID, nhãn thật, dự đoán, 10 xác suất và hash checkpoint. Hash nhận diện file, không đo chất lượng.

## 6. plots.py và utils.py

Plots vẽ ảnh mẫu, đường cong từ history, ma trận từ predictions và feature maps. Khi vẽ ảnh, hoàn tác Normalize bằng `image * std + mean`. Feature map là phản ứng bộ lọc, không chứng minh kênh nào hiểu tay áo. Mỗi kênh có thang màu riêng, không so độ sáng tuyệt đối giữa chúng.

Utils gom việc phụ trợ: đặt seed, lưu JSON và tính SHA256. JSON chứa thông tin có cấu trúc, CSV chứa bảng, NPZ chứa mảng, PT chứa checkpoint. Các file cùng run ID thuộc một lần chạy.

## 7. main.py: nối các phần

`main()` đọc tùy chọn lệnh, tạo thư mục run mới, lưu cấu hình, gọi chuẩn bị dữ liệu. Nếu có `--inspect`, chương trình dừng sau xem dữ liệu. Nếu không, nó train MLP rồi CNN. Chỉ cờ `--evaluate-test` mới thêm bước test. Mỗi lệnh train tạo run mới, chưa hỗ trợ đánh giá độc lập run cũ.

`if __name__ == '__main__':` chỉ gọi main khi chạy file trực tiếp. Import module không tự train. `Path(__file__).resolve().parent` tìm thư mục chứa chương trình, tránh phụ thuộc nơi terminal đang mở.

## 8. Bài học nhỏ nên làm ngay

1. Chạy `main.py --inspect` theo README giai đoạn 3. Tìm 54000/6000, shape ảnh [128,1,28,28], nhãn [128]. Mở train_samples.png trong run vừa in. Giải thích từng chiều trước khi đi tiếp.
2. Đọc models.py, viết lại shape CNN trên giấy. Vì sao 32×7×7 = 1568? Vì sao đầu ra là 10?
3. Đọc run_epoch, chỉ ra backward/step và lý do validation bỏ hai dòng đó. Sau đó chạy smoke để xem vòng học thực sự hoạt động.
4. Sao chép cấu hình smoke rồi thử batch size 32. Dự đoán số batch trước khi chạy. Đây là thực hành kỹ thuật, không thí nghiệm chất lượng.
5. Khi đọc được toàn bộ luồng mới train baseline và lên kế hoạch thay từng yếu tố BN/Dropout/augmentation. Không điều chỉnh cấu hình theo test.

## Tra cú pháp Python

| Cú pháp | Nghĩa |
|---|---|
| `def f(x):` / `return y` | Định nghĩa hàm / trả kết quả; định nghĩa chưa gọi hàm |
| `from .data import make_loader` | Lấy hàm từ module data cùng package |
| `for x in items:` | Lặp, thụt đầu dòng đánh dấu khối |
| `with torch.no_grad():` | Tắt ghi đạo hàm trong khối |
| `None` | Không có đối tượng |
| `Path / 'name'` | Ghép đường dẫn, không chia số |
| `.item()` | Lấy một số từ tensor một phần tử |
| `.cpu().numpy()` | Đưa tensor về CPU rồi chuyển thành mảng NumPy |

Nguồn: notebook tự xây giai đoạn 2. Hai class mô hình được giữ nguyên; các hàm được sửa để truyền tham số rõ ràng. Không chép kết quả nhóm trước thành kết quả mới.
