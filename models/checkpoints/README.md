# Checkpoint dùng cho demo

Các file `.pt` giữ local, không train lại để tạo chúng. Nguồn: `experiments/runs/selection_20261104_144952_e58647/`.

- `mlp.pt`: `mlp/checkpoints/MLP_best.pt`.
- `cnn.pt`: `cnn/checkpoints/CNN_best.pt`.
- `cnn_plus.pt`: `cnn_plus_aug/checkpoints/CNNPlus_best.pt` — bản demo chính.

File JSON cùng tên lưu nhãn và mean/std từ train. `../registry.json` ghi nguồn. Khi lấy ZIP kết quả ở máy khác, copy đúng checkpoint vào đây; không thay bằng trọng số khác mà giữ nguyên tên/metadata.
