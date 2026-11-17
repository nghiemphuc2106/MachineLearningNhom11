# Bài Tập Lớn Học Phần Nhập Môn Học Máy (FIT4013)
## Đề tài 13: Dự đoán Nhân viên có Khả năng Nghỉ việc (HR Employee Attrition Prediction)

- **Trường / Khoa**: Trường Đại học Đại Nam — Khoa Công nghệ thông tin
- **Học phần**: Nhập môn Học máy (`FIT4013`) — Năm học 2026–2027
- **Nhóm thực hiện**: Nhóm 11 (Đề tài 13)
- **Giảng viên hướng dẫn**: ThS. Lê Thị Thùy Trang
- **Bộ dữ liệu**: *IBM HR Analytics Employee Attrition & Performance* (`WA_Fn-UseC_-HR-Employee-Attrition.csv`, gồm `1,470` mẫu nhân viên và `35` thuộc tính; nguồn gốc từ [Kaggle / IBM Watson Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)).

---

## 1. Giới thiệu Bài toán & Điểm nổi bật của Dự án

Dự án xây dựng quy trình học máy (*Machine Learning Pipeline*) hoàn chỉnh từ dữ liệu thô đến đánh giá, phân tích lỗi và ứng dụng thực tế nhằm dự đoán biến mục tiêu nhị phân **`Attrition`** (`Yes` = Nghỉ việc chiếm **16.12%**, `No` = Ở lại chiếm **83.88%**):

1. **Quy trình Học máy Chuẩn mực & Chống Rò rỉ Dữ liệu (*Anti-Data Leakage*)**:
   - Chia dữ liệu phân tầng `Stratified Train/Test Split` theo tỷ lệ **80% Huấn luyện (`1,176` mẫu) / 20% Kiểm thử (`294` mẫu)** với `random_state = 42`.
   - Đóng gói toàn bộ các bước tạo đặc trưng, chuẩn hóa (`StandardScaler` / `RobustScaler`), mã hóa (`OneHotEncoder`) và chọn lọc đặc trưng (`SelectKBest`) vào `sklearn.pipeline.Pipeline`, đảm bảo chỉ gọi `fit()` trên tập Train.
2. **Thiết kế và Đối sánh 2 Luồng Tiền xử lý & Kỹ thuật Đặc trưng (`Luồng A` vs. `Luồng B`)**:
   - **Luồng A (Pipeline cơ bản)**: Loại bỏ 3 cột hằng số (`EmployeeCount`, `Over18`, `StandardHours`) và 1 cột định danh (`EmployeeNumber`), mã hóa `OneHotEncoder(drop='first')` và chuẩn hóa `StandardScaler` (`44` chiều).
   - **Luồng B (Pipeline cải tiến)**:
     - Thiết kế class `HRFeatureEngineer` tạo **8 đặc trưng nghiệp vụ nhân sự mới**: `OverTime_Single_Risk`, `OverTime_LowSat_Risk`, `JobHoppingRate`, `NewHireRisk`, `TotalSatisfaction`, `CommuteBurden`, `IncomePerWorkingYear`, `StagnationRatio`.
     - Loại bỏ thêm 3 cột lương ngẫu nhiên gây nhiễu khoảng cách (`DailyRate`, `HourlyRate`, `MonthlyRate`) và 2 cột đa cộng tuyến mạnh (`JobLevel` có $r=0.95$ với `MonthlyIncome`, `PercentSalaryHike` có $r=0.77$ với `PerformanceRating`).
     - Chuẩn hóa kháng ngoại lệ bằng `RobustScaler` kết hợp chọn lọc đặc trưng `SelectKBest(f_classif, k=15)`.
3. **So sánh Hệ thống các Thuật toán Học máy & Kỹ thuật Mở rộng**:
   - Ba thuật toán cốt lõi của học phần: **K-Nearest Neighbors (K-NN)**, **Gaussian Naive Bayes (GNB)**, **Decision Tree (Cây quyết định)**.
   - Kỹ thuật mở rộng (+0.5 điểm thưởng): **Random Forest Classifier (Ensemble Learning - 200 cây)**.
4. **Ứng dụng Web Demo Tương tác (+0.5 đến +1.0 điểm thưởng)**:
   - Giao diện Web (`app.py`) xây dựng bằng Flask cho phép nhập hồ sơ nhân viên (hoặc chọn nhanh 3 mẫu nhân sự điển hình), đối chiếu xác suất dự đoán của cả 4 mô hình theo thời gian thực và giải thích các chỉ số đặc trưng Luồng B.

---

## 2. Cấu trúc Thư mục Dự án

```text
Machine Learning/
├── WA_Fn-UseC_-HR-Employee-Attrition.csv    # Bộ dữ liệu gốc IBM HR Analytics (1,470 mẫu x 35 cột)
├── src/
│   ├── __init__.py                          # Khởi tạo package src
│   └── pipeline.py                          # Định nghĩa HRFeatureEngineer, hàm chia dữ liệu & Luồng A/B
├── notebooks/
│   └── HR_Employee_Attrition_Pipeline.ipynb # Jupyter Notebook trình bày EDA, Luồng A/B, Mô hình & Phân tích lỗi
├── outputs/
│   ├── figures/                             # 9 biểu đồ trực quan hóa EDA & Đánh giá mô hình (PNG, 200 DPI)
│   ├── models/                              # 4 pipeline mô hình đã huấn luyện (.pkl) phục vụ Web Demo
│   └── results/                             # Bảng kết quả thực nghiệm (CSV/JSON) & các bản ghi lỗi FN/FP
├── Template_BCTT_DATN-main/                 # Mã nguồn Báo cáo LaTeX chuẩn mẫu Trường Đại học Đại Nam (XeLaTeX)
│   ├── main.tex                             # File gốc biên dịch báo cáo
│   ├── settings.tex                         # Cấu hình định dạng (Times New Roman 13pt, giãn dòng 1.5, chương 16pt)
│   ├── cover.tex                            # Hai trang bìa Nhóm 11 chuẩn khung viền Đại học Đại Nam
│   ├── fonts/                               # Bộ font Times New Roman (.ttf) nhúng sẵn cho Overleaf
│   ├── images/                              # Logo trường và 9 biểu đồ thực nghiệm của Đề tài 13
│   ├── frontmatter/                         # Danh mục từ viết tắt (danh_muc_viet_tat.tex) & Lời mở đầu (mo_dau.tex)
│   ├── chapters/                            # Nội dung 3 chương chính (chuong1.tex, chuong2.tex, chuong3.tex)
│   └── backmatter/                          # Kết luận (ket_luan.tex), Phụ lục (phu_luc.tex), Tài liệu tham khảo
├── BaoCaoNhom11.docx                        # Bản Báo cáo Word (.docx) hoàn chỉnh của Nhóm 11
├── run_experiments.py                       # Script chạy tự động toàn bộ thí nghiệm, xuất biểu đồ & lưu mô hình
├── app.py                                   # Ứng dụng Web Demo dự đoán nghỉ việc tương tác (Flask)
├── requirements.txt                         # Danh sách thư viện Python và phiên bản
├── AI_USAGE.md                              # Tài liệu khai báo minh bạch việc sử dụng công cụ AI
├── .gitignore                               # Loại trừ cache Python, Matplotlib và file tạm Word khi đẩy lên Git
└── README.md                                # Tài liệu hướng dẫn sử dụng và tái lập kết quả
```

---

## 3. Kết quả Thực nghiệm Chính (Trên tập Test độc lập, N = 294)

| Luồng xử lý | Mô hình & Cấu hình | CV F1 (Train) | Test Accuracy | Precision (Yes) | Recall (Yes) | **Test F1** | **Test ROC-AUC** | TP / FN | FP / TN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Luồng A (Cơ bản)** | K-NN (Baseline, `k=5`, L2) | 0.2264 | 83.67% | 45.45% | 10.64% | 0.1724 | 0.6181 | 5 / 42 | 6 / 241 |
| **Luồng A (Cơ bản)** | K-NN (Tuned, `k=11`, L1, dist) | 0.1929 | 84.01% | 50.00% | 6.38% | 0.1132 | 0.6921 | 3 / 44 | 3 / 244 |
| **Luồng A (Cơ bản)** | Gaussian NB (Baseline, `1e-9`) | 0.3902 | 61.56% | 25.00% | **70.21%** | 0.3687 | 0.6956 | 33 / 14 | 99 / 148 |
| **Luồng A (Cơ bản)** | Gaussian NB (Tuned, `smooth=0.1`) | 0.4745 | 74.49% | 32.93% | 57.45% | 0.4186 | 0.7324 | 27 / 20 | 55 / 192 |
| **Luồng A (Cơ bản)** | Decision Tree (Baseline, mặc định) | 0.3714 | 75.85% | 29.31% | 36.17% | 0.3238 | 0.5979 | 17 / 30 | 41 / 206 |
| **Luồng A (Cơ bản)** | Decision Tree (Tuned, `depth=4, bal.`) | 0.4217 | 70.07% | 26.97% | 51.06% | 0.3529 | 0.6960 | 24 / 23 | 65 / 182 |
| **Luồng B (Cải tiến)** | K-NN (Baseline, `k=5`, L2) | 0.3647 | 83.33% | 43.75% | 14.89% | 0.2222 | 0.6596 | 7 / 40 | 9 / 238 |
| **Luồng B (Cải tiến)** | K-NN (Tuned, `k=7`, L1, dist) | 0.3181 | 85.03% | **60.00%** | 19.15% | 0.2903 | 0.6977 | 9 / 38 | 6 / 241 |
| **Luồng B (Cải tiến)** | Gaussian NB (Baseline, `1e-9`) | 0.4874 | 80.27% | 42.47% | 65.96% | 0.5167 | **0.7701** | 31 / 16 | 42 / 205 |
| **Luồng B (Cải tiến)** | **Gaussian NB (Tuned, `smooth=0.1`)** | **0.5411** | **85.71%** | **54.90%** | **59.57%** | **0.5714** | 0.7686 | **28 / 19** | **23 / 224** |
| **Luồng B (Cải tiến)** | Decision Tree (Baseline, mặc định) | 0.3439 | 75.85% | 26.00% | 27.66% | 0.2680 | 0.5634 | 13 / 34 | 37 / 210 |
| **Luồng B (Cải tiến)** | Decision Tree (Tuned, `depth=4, bal.`) | 0.4092 | 75.17% | 33.33% | 55.32% | 0.4160 | 0.7033 | 26 / 21 | 52 / 195 |
| **Mở rộng (Ensemble)** | Random Forest (`200 cây, depth=7, bal.`) | 0.5033 | 76.53% | 35.53% | 57.45% | 0.4390 | 0.7607 | 27 / 20 | 49 / 198 |

---

## 4. Hướng dẫn Cài đặt, Chạy Chương trình và Xuất Báo cáo

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Tái lập toàn bộ thực nghiệm, biểu đồ và mô hình
```bash
python run_experiments.py
```
Lệnh trên sẽ tự động:
- Xuất 9 biểu đồ chất lượng cao (`200 DPI`) vào `outputs/figures/`.
- Xuất bảng kết quả thực nghiệm (`experiment_summary.csv`, `.json`) và danh sách mẫu lỗi (`error_analysis_FN.csv`, `FP.csv`) vào `outputs/results/`.
- Lưu 4 pipeline mô hình đã huấn luyện vào `outputs/models/`.

### Bước 3: Khởi chạy Ứng dụng Web Demo
```bash
python app.py
```
Sau đó mở trình duyệt truy cập: `http://127.0.0.1:5000`

### Bước 4: Xem Notebook, Báo cáo Word hoặc Biên dịch Báo cáo LaTeX
- **Jupyter Notebook**: Mở tệp `notebooks/HR_Employee_Attrition_Pipeline.ipynb` bằng VS Code hoặc JupyterLab.
- **Báo cáo Word (`.docx`)**: Mở trực tiếp tệp `BaoCaoNhom11.docx` bằng Microsoft Word.
- **Báo cáo LaTeX (Mẫu Trường Đại học Đại Nam)**:
  - **Biên dịch cục bộ (TeX Live / MiKTeX)**: Mở thư mục `Template_BCTT_DATN-main/` và chạy `xelatex main.tex` (2 lần để cập nhật mục lục và danh mục bảng/hình).
  - **Biên dịch trên Overleaf**: Nén thư mục `Template_BCTT_DATN-main/` thành `.zip` rồi tải lên Overleaf, chọn trình biên dịch **XeLaTeX** và bấm **Recompile** để xuất file PDF.

---

## 5. Thành viên Thực hiện (Nhóm 11 — Đề tài 13)

| STT | Mã sinh viên | Họ và Tên | Ngày sinh |
| :---: | :---: | :--- | :---: |
| 1 | `1871020457` | Nghiêm Minh Phúc | 21/06/2006 |
| 2 | `1871020290` | Trần Phạm Gia Huy | 03/11/2006 |
| 3 | `1871020103` | Nguyễn Tiến Cường | 07/12/2006 |
| 4 | `1871020099` | Nguyễn Thành Công | 24/04/2006 |
