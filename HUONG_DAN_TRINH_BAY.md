# Kịch bản trình diễn khoảng 5–7 phút

1. Tổng quan: nêu bài toán phân loại nhị phân từ 6 thông số, AI4I là dữ liệu mô phỏng. Chia train/validation/test 7.000/1.500/1.500.
2. Dự đoán một mẫu: chọn mẫu không hỏng, bấm Phân tích; tiếp tục mẫu có hỏng. Hai mẫu validation đã chọn đúng để minh họa thao tác, không chứng minh độ chính xác chung.
3. Sửa nhiệt độ thành -1 rồi phân tích: chương trình từ chối. Sửa thành giá trị dương ngoài train: chương trình cảnh báo, không kết luận máy chắc chắn hỏng vì vượt khoảng.
4. CSV: tải mẫu và tải lại lên; xem kết quả từng dòng và tải CSV. File gốc 10.000 mẫu có train nên không dùng để báo cáo hiệu quả kiểm thử độc lập.
5. Kiểm chứng test: bấm chạy. Chỉ ra TP38, FN13, FP16, TN1433. Lọc Bỏ sót và Cảnh báo nhầm để thể hiện giới hạn thực tế của mô hình.
6. So sánh: LightGBM được chọn theo AP validation; ngưỡng theo F1 validation. Không chọn theo điểm test; nâng cao không mặc định tốt hơn.
7. Giải thích SHAP: màu cam tăng điểm hỏng, màu xanh giảm; đây là đóng góp vào quyết định mô hình, không phải chứng minh nguyên nhân vật lý.

## Gắn với báo cáo

- Thu thập/tiền xử lý: nguồn UCI, thống kê chất lượng, 6 đầu vào, loại cột rò rỉ, chia dữ liệu và pipeline.
- Phương pháp: bốn thuật toán, cấu hình, CV, tiêu chí AP, chọn ngưỡng F1, thử nghiệm hiệu chỉnh.
- Kết quả: reports/validation_comparison.csv, metrics.csv, pr_curve.png, confusion_test.png; nêu cả bỏ sót và cảnh báo nhầm.
- Demo: ảnh giao diện Tổng quan, Dự đoán, Kiểm chứng test; dẫn link web sau khi triển khai.
- Hạn chế: mô phỏng, chưa kết nối máy thật, test đã được quan sát trong phát triển, lịch sử chỉ lưu phiên.
