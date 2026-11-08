# Kiểm tra

- `python -m tests.kiem_tra_du_an`: kiểm tra cấu trúc, dữ liệu, checkpoint và inference; được chạy trên máy cá nhân, không optimizer.
- `python -m tests.kiem_tra_mo_hinh --output reports/verification/20261108/kiem_tra_mo_hinh.json`: kiểm tra split, thống kê chuẩn hóa, số liệu và chạy suy luận trên toàn bộ 10.000 ảnh test cho ba checkpoint demo. Chạy CPU, không cập nhật trọng số. Nếu kiểm tra vào ngày khác, đổi thư mục đầu ra để giữ biên bản cũ.
- `python -m tests.kiem_tra_huan_luyen_gpu`: kiểm tra vòng học/dừng sớm/scheduler/nạp checkpoint, chỉ Colab GPU.
- `python -m unittest tests.test_quy_trinh_gpu`: kiểm tra pipeline lịch sử, chỉ Colab GPU.

Không dùng kết quả tensor giả để báo accuracy đồ án. Các kiểm tra train đã chạy trước khi đổi cấu trúc; sau đổi mới kiểm tra tĩnh và inference, chưa train lại.

[Biên bản kiểm tra ngày 08/11/2026](../reports/verification/20261108/bien_ban_kiem_tra.md) ghi các kiểm tra đã chạy và phạm vi còn giới hạn.
