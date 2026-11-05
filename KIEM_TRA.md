# Kiểm tra bản làm lại — 04/11/2026

## Đã thực hiện

- Đọc lại gói chương trình cũ và kiểm tra mã học máy.
- So SHA-256 CSV gốc với thông tin thực nghiệm: khớp.
- Huấn luyện lại 8 cấu hình, tìm siêu tham số LightGBM 12 × 3 folds, thử hiệu chỉnh sigmoid 3-fold CV train. Lưu nhật ký thực tế trong reports/training_log.txt.
- Chọn lại LightGBM chuẩn và ngưỡng 0,266037; tái lập TN1433, FP16, FN13, TP38 trên test. Không ép kết quả mới về giá trị cũ.
- 12 kiểm tra gồm 6 kiểm tra lõi, 3 kiểm tra dữ liệu/mô hình và 3 kịch bản AppTest.
- Kịch bản AppTest mở đủ 7 trang, chọn mẫu hỏng, chạy dự đoán, kiểm chứng 1.500 test, lọc bỏ sót, kiểm tra lịch sử và xóa lịch sử.
- Kiểm tra split 7.000/1.500/1.500 không giao nhau; min/max lấy từ train.
- Đổi cột nhãn không làm đổi dự đoán; xác suất web khớp thực nghiệm theo source_row.
- Kiểm tra TreeSHAP trên toàn bộ test: tổng đóng góp + nền khớp điểm thô; sigmoid điểm thô khớp xác suất mô hình nền với dung sai 1e-10.

## Chưa xác nhận

- Chưa triển khai bản mới lên tài khoản Streamlit của người dùng.
- Giao diện được kiểm tra bằng AppTest, chưa kiểm tra ảnh chụp trình duyệt thực hoặc điện thoại. Tải trình duyệt kiểm thử không thành công trong môi trường xây dựng.
- Luồng tải file thật qua trình duyệt và nút tải xuống cần thử một lần sau triển khai; lõi xử lý CSV được kiểm tra bằng dữ liệu thật và dữ liệu lỗi.
- Đây không phải kiểm định trên máy công nghiệp thật.

## Kiểm tra nhanh sau triển khai

1. Mở đủ 7 trang không báo lỗi.
2. Mẫu validation không hỏng trả khoảng 0,16%; mẫu hỏng khoảng 93,51%.
3. Nhập số âm → báo lỗi; số dương ngoài phạm vi → báo “Không đủ cơ sở kết luận”.
4. Tải data/mau_nhap.csv lên CSV hàng loạt; tải kết quả xuống và mở được.
5. Kiểm chứng test → 1.500 mẫu, TP38/FN13/FP16/TN1433, thông báo khớp kết quả đã lưu.
6. Xem biểu đồ giải thích trên màn hình trình chiếu, tên thuộc tính đọc được.
