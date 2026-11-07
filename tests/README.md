# Kiểm tra

- `python -m tests.verify_layout`: kiểm tra cấu trúc, dữ liệu, checkpoint và inference; được chạy trên máy cá nhân, không optimizer.
- `python -m tests.verify_training_gpu`: kiểm tra vòng học/dừng sớm/scheduler/nạp checkpoint, chỉ Colab GPU.
- `python -m unittest tests.test_pipeline_gpu`: kiểm tra pipeline lịch sử, chỉ Colab GPU.

Không dùng kết quả tensor giả để báo accuracy đồ án. Các kiểm tra train đã chạy trước khi đổi cấu trúc; sau đổi mới kiểm tra tĩnh và inference, chưa train lại.
