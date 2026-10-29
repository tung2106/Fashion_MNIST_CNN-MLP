# Phân loại trang phục — bản đồ dự án

Cập nhật tổ chức thư mục: 29/10/2026. Bắt đầu bằng tài liệu này khi mở dự án trong VS Code.

## Tìm đúng nơi làm việc

| Thư mục | Chứa gì / dùng để làm gì | Trạng thái |
|---|---|---|
| [TAI_LIEU](TAI_LIEU/README.md) | Hai file đặc tả/kỹ năng, kế hoạch 14 ngày và phân công 5 người, báo cáo tham khảo, hướng dẫn Colab, báo cáo nhóm | Đã tập hợp tài liệu hiện có |
| [01_Lam_Quen_Du_Lieu](01_Lam_Quen_Du_Lieu/README.md) | Bài học đọc ảnh, pixel, nhãn, đếm số ảnh | Có mã thực hành |
| [02_Colab_Co_Ban](02_Colab_Co_Ban/README.md) | Notebook MLP/CNN cơ bản, mã nguồn, kết quả train/test, checkpoint và biểu đồ | Đã có một lần chạy thực tế |
| [03_Thuc_Nghiem_Cai_Tien](03_Thuc_Nghiem_Cai_Tien/README.md) | Mã và kết quả thí nghiệm có kiểm soát, cải tiến mô hình, phân tích đặc trưng | Chưa triển khai |
| [04_Demo_Truc_Quan](04_Demo_Truc_Quan/README.md) | Giao diện demo dự đoán và giải thích kết quả | Chưa triển khai ứng dụng riêng |
| [05_Bao_Cao_Va_Ban_Giao](05_Bao_Cao_Va_Ban_Giao/README.md) | Công cụ kiểm tra, đóng gói và bản bàn giao cuối | Chưa triển khai |
| [DU_LIEU](DU_LIEU/README.md) | Dữ liệu Fashion-MNIST gốc dùng chung trên máy | Đã có ảnh và nhãn train/test |
| [RAC](RAC/README.md) | Gói hướng dẫn cũ và tệp tạm trích xuất báo cáo | Giữ lại để có thể khôi phục |
| `.vscode` | Cấu hình trình soạn thảo | Giữ tại gốc để VS Code nhận diện |

Tài liệu nội dung được tập trung trong `TAI_LIEU`; README tại từng giai đoạn chỉ là bản chỉ đường và danh sách việc của giai đoạn đó. Các thư mục giai đoạn 03–05 hiện có hướng dẫn, chưa có mã/kết quả hoàn thành.

## Thứ tự học và thực hiện

1. **Hiểu dữ liệu:** mở [xem_du_lieu.py](01_Lam_Quen_Du_Lieu/xem_du_lieu.py), xem một ảnh và nhãn. Đây là tệp `test.py` cũ đã đổi tên.
2. **Hiểu vòng học:** mở [notebook cơ bản](02_Colab_Co_Ban/Fashion_MNIST_Hoc_Tung_Buoc.ipynb), đọc từng ô theo [hướng dẫn Colab](TAI_LIEU/03_Huong_Dan_Colab/HUONG_DAN_COLAB.md).
3. **Thí nghiệm cải tiến:** theo [đặc tả và kế hoạch](TAI_LIEU/01_Dac_Ta_Va_Ke_Hoach/01_DAC_TA_VA_PROMPT.md) và [kỹ năng triển khai](TAI_LIEU/01_Dac_Ta_Va_Ke_Hoach/02_KY_NANG_TRIEN_KHAI.md). Lưu mã/kết quả mới vào giai đoạn 03, giữ riêng lần chạy cơ bản.
4. **Làm demo:** dùng checkpoint đã được lựa chọn qua validation, tái sử dụng xử lý ảnh và mô hình của giai đoạn 03.
5. **Viết và bàn giao:** bản báo cáo/slide của nhóm vào `TAI_LIEU/04_Bao_Cao_Nhom`; gói nộp cuối vào giai đoạn 05.

Kế hoạch hai tuần và phân công nằm trong hai file đặc tả, vẫn là nguồn kế hoạch chính. Tên giai đoạn chỉ thể hiện trình tự và trách nhiệm lưu trữ; thành viên có thể chuẩn bị tài liệu/demo song song với thí nghiệm.

## Kết quả đã có

Lần chạy `20261029_092231_60aa1b` ở giai đoạn 02: 5 epoch mỗi mô hình, train/validation/test = **54.000/6.000/10.000 ảnh**, tổng **70.000 ảnh**. Accuracy test: MLP **87,50%**, CNN **90,27%**. Đây là thực hành cơ bản, chưa hoàn thành toàn bộ yêu cầu đồ án.

- [Trạng thái và giới hạn kết quả](TAI_LIEU/03_Huong_Dan_Colab/TRANG_THAI_THUC_HIEN.md)
- [Notebook đã chạy trên Colab](https://colab.research.google.com/drive/1himC9AFXIiUB5KNYW5PlEPnTs7OaOtvL)
- [Bảng kết quả cục bộ](02_Colab_Co_Ban/ket_qua/20261029_092231_60aa1b/metrics/summary.csv)

Đổi thư mục cục bộ không đổi đường dẫn notebook trên Google Colab. Đường dẫn `/content/fashion_mnist_basic` trong notebook là trên máy Colab, được giữ nguyên.

## Quy tắc lưu tệp mới

- Ảnh/nhãn gốc dùng chung → `DU_LIEU`; mã và kết quả thực hiện → đúng thư mục giai đoạn.
- Mỗi lần thí nghiệm có thư mục kết quả riêng theo run ID; giữ cấu hình, checkpoint, dự đoán, chỉ số và hình cùng nhau.
- Tài liệu hướng dẫn, báo cáo, slide → `TAI_LIEU`; ảnh minh họa tạo từ thí nghiệm ở cùng run nguồn, chỉ sao chép vào báo cáo khi cần.
- Tệp cũ/tạm → `RAC` kèm lý do. Không coi checkpoint, dữ liệu `.gz` hay gói kết quả ZIP là rác chỉ vì có bản giải nén.

Không xóa tệp gốc trong lần sắp xếp này. [Nhật ký chuyển tệp](TAI_LIEU/NHAT_KY_SAP_XEP.json) lưu đường dẫn cũ/mới và mã SHA256 trước chỉnh sửa để đối chiếu.
