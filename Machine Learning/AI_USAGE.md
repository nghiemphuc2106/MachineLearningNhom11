# Khai báo Sử dụng Công cụ AI (AI Usage Declaration)

Tài liệu này được lập theo đúng quy định tại **Mục 2.3 (Hình thức báo cáo và nộp bài)** và **Mục 2.5 (Rubric đánh giá bài tập lớn A3 - Tiêu chí Liêm chính học thuật và sử dụng AI)** của học phần **Nhập môn Học máy (FIT4013)**.

## 1. Công cụ AI đã sử dụng

- **Tên công cụ**: Trợ lý lập trình AI (Gemini).
- **Thời gian sử dụng**: Trong quá trình thiết kế kiến trúc mã nguồn, xây dựng ứng dụng Web Demo và định dạng báo cáo khoa học LaTeX.

## 2. Phạm vi và Mục đích sử dụng AI

Nhóm sử dụng công cụ AI với vai trò hỗ trợ kỹ thuật ở các khâu sau:

1. **Tổ chức và chuẩn hóa mã nguồn (`src/pipeline.py`, `run_experiments.py`)**:
   - Hỗ trợ đóng gói quy trình tiền xử lý Luồng A và Luồng B vào `sklearn.pipeline.Pipeline` và `ColumnTransformer` nhằm đảm bảo không xảy ra rò rỉ dữ liệu (*data leakage*) giữa tập Train và tập Test.
   - Gợi ý cách viết class `HRFeatureEngineer` kế thừa `BaseEstimator` và `TransformerMixin` đúng chuẩn `scikit-learn`.
2. **Xây dựng giao diện Web App Demo (`app.py`)**:
   - Hỗ trợ thiết kế giao diện HTML/CSS và REST API bằng `Flask` để tải các mô hình đã huấn luyện (`outputs/models/*.pkl`) và hiển thị trực quan xác suất nghỉ việc kèm các đặc trưng nghiệp vụ nhân sự.
3. **Soạn thảo và định dạng LaTeX (`report/Bao_Cao_BTL_FIT4013_DeTai13.tex`)**:
   - Hỗ trợ tạo khung định dạng bảng biểu (`booktabs`), sơ đồ khối (`TikZ`) và kiểm tra lỗi biên dịch LaTeX tiếng Việt.

## 3. Phần công việc do Nhóm kiểm chứng, thực nghiệm và chịu trách nhiệm

- **Dữ liệu và Kết quả thực nghiệm**: Toàn bộ dữ liệu đầu vào (`WA_Fn-UseC_-HR-Employee-Attrition.csv`), quá trình chia tập dữ liệu (`test_size=0.2, stratify=y, random_state=42`), kết quả kiểm định chéo 5-Fold CV và các chỉ số trên tập Test (`Accuracy`, `Precision`, `Recall`, `F1-score`, `ROC-AUC`, `Confusion Matrix`) đều được chạy thực nghiệm trực tiếp và lưu minh chứng tại `outputs/results/experiment_summary.csv`.
- **Phân tích chuyên môn và Giải trình**: Các giả thuyết về kỹ thuật đặc trưng (8 đặc trưng HR mới), lý giải nguyên nhân K-NN bị ảnh hưởng bởi mất cân bằng lớp, lý giải vai trò của tham số làm trơn phương sai `var_smoothing` trong Gaussian Naive Bayes, và phân tích từng bản ghi dự đoán sai (`error_analysis_FN.csv`, `error_analysis_FP.csv`) đều được kiểm tra đối chiếu với dữ liệu gốc.
- **Cam kết của các thành viên**: Toàn bộ thành viên trong nhóm đã đọc hiểu, chạy thử nghiệm độc lập và nắm vững toàn bộ pipeline, mã nguồn cũng như kết quả để bảo vệ trước giảng viên.
