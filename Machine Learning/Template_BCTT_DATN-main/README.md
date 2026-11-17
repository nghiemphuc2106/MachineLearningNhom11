# Báo cáo Bài tập lớn Học phần Nhập môn Học máy (FIT4013)
## Đề tài 13: Dự đoán Nhân viên có Khả năng Nghỉ việc (HR Employee Attrition)

Thư mục này chứa toàn bộ mã nguồn **LaTeX (XeLaTeX)** của báo cáo bài tập lớn Đề tài 13, được chuẩn hóa theo mẫu của Khoa Công nghệ thông tin — Trường Đại học Đại Nam.

## 1. Cấu trúc thư mục

```text
Template_BCTT_DATN-main/
├── main.tex                       # File chính điều khiển cấu trúc toàn bộ báo cáo
├── settings.tex                   # Cấu hình định dạng (Times New Roman 13pt, giãn dòng 1.5, chương 16pt IN HOA)
├── cover.tex                      # Hai trang bìa có khung viền theo mẫu Trường Đại học Đại Nam
├── fonts/                         # Nhúng sẵn 4 file font Times New Roman chuẩn (.ttf)
├── images/                        # Chứa logo-dai-nam.png và 9 biểu đồ thực nghiệm (fig1 -> fig9.png)
├── frontmatter/
│   ├── danh_muc_viet_tat.tex      # Danh mục các từ viết tắt
│   └── mo_dau.tex                 # Lời mở đầu
├── chapters/
│   ├── chuong1.tex                # Chương 1: Đặt vấn đề, phân tích yêu cầu và khám phá dữ liệu (EDA)
│   ├── chuong2.tex                # Chương 2: Thiết kế giải pháp, quy trình học máy và phân tích mã nguồn
│   └── chuong3.tex                # Chương 3: Thử nghiệm, đánh giá kết quả, thảo luận và phân tích lỗi
└── backmatter/
    ├── ket_luan.tex               # Kết luận và hướng phát triển
    ├── phu_luc.tex                # Phụ lục: Hướng dẫn cài đặt và chạy chương trình
    └── tai_lieu_tham_khao.tex     # Danh mục tài liệu tham khảo
```

## 2. Hướng dẫn chỉnh sửa thông tin cá nhân và biên dịch

1. **Sửa thông tin sinh viên/nhóm trên trang bìa**:
   - Thông tin chung (tên nhóm, học phần, giảng viên hướng dẫn) khai báo tại `settings.tex` (Mục 8):
     ```latex
     \newcommand{\TenSinhVien}{Nhóm 11 (Đề tài 13)}
     \newcommand{\MaSinhVien}{FIT4013 -- Nhóm 11}
     \newcommand{\KhoaHoc}{2026 -- 2027}
     ```
   - Danh sách 4 thành viên nhóm trên trang bìa 2 được định nghĩa trực tiếp trong `cover.tex`.

2. **Biên dịch sang PDF**:
   - **Trên Overleaf**: Nén thư mục `Template_BCTT_DATN-main/` thành file `.zip` rồi tải lên Overleaf, chọn trình biên dịch **XeLaTeX** và nhấn **Recompile**.
   - **Trên máy tính cá nhân (đã cài TeX Live / MiKTeX)**:
     ```bash
     xelatex main.tex
     xelatex main.tex
     ```
     *(Chạy 2 lần để LaTeX cập nhật tự động số trang cho Danh mục Bảng, Sơ đồ, Hình và Mục lục).*
