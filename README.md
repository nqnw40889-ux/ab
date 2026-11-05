# NHÓM 16 — MACHINE HEALTH · Bản làm lại

Ứng dụng phân loại trạng thái hỏng/không hỏng trên dữ liệu AI4I 2020 mô phỏng.
Bản này có mô hình đã huấn luyện, dữ liệu gốc, 3 tập chia, mã tái lập và trang kiểm chứng test.

## 1. Chạy trên máy

Dùng Python 3.12 (bản kiểm thử: 3.12.14). Mở terminal ở thư mục chứa app.py:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mở địa chỉ localhost hiện trong terminal. Không cần chạy huấn luyện để mở web.
Windows có thể dùng `py -3.12 -m venv .venv`, rồi `.venv\Scripts\activate` trước khi cài.

## 2. Đưa lên Streamlit Community Cloud

1. Giải nén gói ZIP. Tải **nội dung đã giải nén** lên GitHub; không chỉ tải file ZIP.
2. Giữ app.py, core.py, evaluation.py, requirements.txt, mo_hinh_demo.joblib ở cùng cấp. Giữ nguyên thư mục data/, reports/ và .streamlit/.
3. Khi tạo ứng dụng, chọn đúng repository, branch chứa mã; Main file path: `app.py` nếu app.py nằm ở gốc repo.
4. Trong Advanced settings chọn Python **3.12** để phù hợp môi trường đã kiểm thử.
5. Triển khai. Khi mở được web, vào Kiểm chứng test, bấm chạy và xác nhận khớp kết quả đã lưu.
6. Nếu báo “This file does not exist”, xem app.py đang nằm trong thư mục con nào và điền đường dẫn đầy đủ, ví dụ `Nhom16_MachineHealth/app.py`.

Bản này chưa được tự động đưa lên tài khoản GitHub/Streamlit của bạn. Bạn chủ động giữ bản đang chạy làm bản dự phòng.

## 3. Các trang

- Tổng quan: đề tài, quy trình, chỉ số test.
- Dự đoán một mẫu: 6 trường, kiểm tra hợp lệ, cảnh báo tự động ngoài min/max train, điểm/ngưỡng, thông số lần dự đoán, TreeSHAP.
- CSV hàng loạt: tối đa 10 MB và 10.000 dòng; giữ dòng lỗi, thông báo lỗi theo dòng, tải kết quả. Tệp UTF-8, phân cách dấu phẩy, số thập phân dấu chấm.
- Kiểm chứng test: thực sự tính lại dự đoán bằng mô hình web trên 1.500 mẫu; so với nhãn thật và xác suất đã lưu; lọc bốn nhóm đúng/sai.
- So sánh mô hình: so sánh trên validation, ma trận nhầm lẫn và đường PR.
- Lịch sử: tối đa 1.000 dự đoán trong phiên; tải CSV trước khi đóng. Không có cơ sở dữ liệu lưu lâu dài.
- Thông tin đề tài: nguồn dữ liệu, đơn vị, nhãn mục tiêu và phạm vi ứng dụng.

## 4. Dữ liệu trong gói

`data/ai4i2020.csv`: gốc UCI, 10.000 mẫu.
`data/train_goc.csv`, `validation_goc.csv`, `test_goc.csv`: 6 đầu vào + source_row + Machine failure; source_row là chỉ số dòng 0-based của dữ liệu gốc, không dùng làm thuộc tính.
`data/mau_nhap.csv`: chỉ 6 đầu vào để thử chức năng CSV.
`reports/*_indices.csv`: danh sách mẫu từng tập; `predictions_*.csv`: nhãn và xác suất thực nghiệm.
Không dùng các cột UDI, Product ID, TWF, HDF, PWF, OSF, RNF làm đầu vào.
Các cột thừa trong CSV được giữ khi xuất nhưng không đưa vào mô hình.

## 5. Huấn luyện lại có thể tái lập

```bash
python train.py --csv data/ai4i2020.csv --output trained_new --advanced
```

Chạy 8 cấu hình thuộc 4 thuật toán. Chuẩn hóa chỉ áp dụng LR; one-hot Type với L tham chiếu, thứ tự M/H cố định. Mỗi pipeline fit trên train; CV fit lại cả pipeline trong từng fold.
Chia phân tầng 70/15/15 với seed42. So sánh AP validation; tìm ngưỡng tối đa F1 validation (nếu bằng nhau chọn ngưỡng cao nhất). Tìm siêu tham số chỉ LightGBM: 12 ứng viên × 3 folds; sigmoid calibration dùng 3-fold CV train, chọn theo Brier validation. Không ép biến thể nâng cao phải tốt hơn.
Chỉ tính test sau khi lưu lựa chọn. Script từ chối ghi đè thư mục đã chứa mô hình.
Để dùng kết quả mới: thay mô hình, metadata, toàn bộ reports/ và data/ cùng nhau từ trained_new. Không chỉ thay mô hình mà giữ báo cáo cũ.
Notebook `Nhom16_HuanLuyen_KiemChung.ipynb` hướng dẫn chạy cùng quy trình trên Colab, không thay notebook tiền xử lý cũ.

## 6. Kết quả chạy lại của gói này

LightGBM không trọng số vẫn được chọn; ngưỡng ~0,266037. Hiệu chỉnh không được chọn.
Test: TN1433, FP16, FN13, TP38. Accuracy 98,0667%; Precision 70,3704%; Recall 74,5098%; F1 0,723810; AP 0,800627.
Không phải cam kết mọi mẫu đều đúng. Tập test đã được quan sát ở các vòng phát triển trước; kết quả tái lập không biến tập này thành một tập xác nhận độc lập mới.

## 7. Kiểm tra chương trình

```bash
python -m unittest discover -s tests -v
```

Kiểm tra đầu vào, ngoài phạm vi, lô có dòng lỗi, tải lại mô hình, split không giao nhau, web khớp kết quả thực nghiệm, nhãn không được đưa vào mô hình, SHAP cộng tính, điều hướng và thao tác giao diện bằng AppTest.
Xem `KIEM_TRA.md` để biết những gì đã thực sự chạy và giới hạn kiểm thử.

## 8. Diễn giải đúng

- Đây là dữ liệu mô phỏng, không phải cảm biến trực tiếp từ máy thật.
- Không có nhãn thật thì chưa xác minh được đúng/sai của mẫu tự nhập.
- Min/max train là khoảng quan sát, không phải giới hạn an toàn. Dòng tự nhập hoặc CSV ngoài khoảng được ghi “Không đủ cơ sở kết luận”, trống nhãn và xác suất. Không tự ép giá trị về min/max.
- Điểm xác suất chưa hiệu chỉnh, không phải xác suất rủi ro đã kiểm chứng trên máy thật.
- TreeSHAP mô tả mô hình, không chứng minh nguyên nhân hỏng. Biểu đồ log-odds có kiểm tra cộng tính với giá trị nền.
- Không dự đoán thời điểm máy hỏng, không xử lý ảnh, không kết nối thiết bị thật.

Nguồn: https://archive.ics.uci.edu/dataset/601/ai4i (CC BY 4.0). Giữ ghi nhận nguồn khi sử dụng lại dữ liệu.
Tài liệu: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest ; https://lightgbm.readthedocs.io/en/stable/Parameters.html
