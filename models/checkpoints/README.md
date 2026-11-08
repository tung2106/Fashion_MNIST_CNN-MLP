# Checkpoint dùng cho demo

Ba file `.pt` được đưa lên GitHub để chạy demo trên máy khác, không cần train lại. Tổng dung lượng khoảng 3,1 MB. Nguồn: `experiments/runs/selection_20261104_144952_e58647/`.

- `mlp.pt`: `mlp/checkpoints/MLP_best.pt`.
- `cnn.pt`: `cnn/checkpoints/CNN_best.pt`.
- `cnn_plus.pt`: `cnn_plus_aug/checkpoints/CNNPlus_best.pt` — bản demo chính.

File JSON cùng tên lưu nhãn và mean/std từ train. `../danh_sach_mo_hinh.json` ghi nguồn và SHA256. Clone repository sẽ có sẵn checkpoint; không thay bằng trọng số khác mà giữ nguyên tên/metadata. Xem `../../demo/README.md` để cài và chạy.
