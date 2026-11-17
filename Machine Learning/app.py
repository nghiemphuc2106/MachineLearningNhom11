import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

# Import custom transformer so joblib can unpickle pipelines cleanly
from src.pipeline import HRFeatureEngineer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOD_DIR = os.path.join(BASE_DIR, "outputs", "models")
RES_DIR = os.path.join(BASE_DIR, "outputs", "results")
CSV_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

app = Flask(__name__)

# Load models and experiment summary
MODELS = {
    "gnb": ("Gaussian Naive Bayes (Tuned - Luồng B) [F1=0.5714, Khuyên dùng]", joblib.load(os.path.join(MOD_DIR, "best_gnb_stream_b.pkl"))),
    "rf": ("Random Forest Ensemble (Mở rộng - Luồng B) [F1=0.4390]", joblib.load(os.path.join(MOD_DIR, "best_rf_extension.pkl"))),
    "dt": ("Decision Tree (Tuned - Luồng B) [F1=0.4160]", joblib.load(os.path.join(MOD_DIR, "best_dt_stream_b.pkl"))),
    "knn": ("K-Nearest Neighbors (Tuned - Luồng B) [F1=0.2903]", joblib.load(os.path.join(MOD_DIR, "best_knn_stream_b.pkl")))
}

with open(os.path.join(RES_DIR, "experiment_summary.json"), "r", encoding="utf-8") as f:
    EXPERIMENT_SUMMARY = json.load(f)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FIT4013 - Đề tài 13: Hệ thống Dự đoán Nhân viên Nghỉ việc</title>
    <style>
        :root {
            --primary: #1e3a8a;
            --primary-light: #2563eb;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #0f172a;
            --muted: #475569;
            --border: #e2e8f0;
            --danger: #dc2626;
            --warning: #d97706;
            --success: #16a34a;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body { background: var(--bg); color: var(--text); line-height: 1.5; padding-bottom: 40px; }
        header {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: white; padding: 24px 32px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        header h1 { font-size: 1.45rem; font-weight: 700; margin-bottom: 4px; }
        header p { color: #cbd5e1; font-size: 0.92rem; }
        .container { max-width: 1280px; margin: 24px auto; padding: 0 20px; display: grid; grid-template-columns: 7fr 5fr; gap: 24px; }
        @media (max-width: 960px) { .container { grid-template-columns: 1fr; } }
        .card {
            background: var(--card); border: 1px solid var(--border); border-radius: 10px;
            padding: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-bottom: 20px;
        }
        .card h2 { font-size: 1.12rem; color: var(--primary); margin-bottom: 14px; border-bottom: 2px solid #eff6ff; padding-bottom: 8px; }
        .preset-bar { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }
        .btn-preset {
            background: #f1f5f9; border: 1px solid #cbd5e1; color: #1e293b;
            padding: 7px 13px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600;
            transition: all 0.15s;
        }
        .btn-preset:hover { background: #e2e8f0; border-color: #94a3b8; }
        .form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        @media (max-width: 680px) { .form-grid { grid-template-columns: 1fr 1fr; } }
        .field label { display: block; font-size: 0.8rem; font-weight: 600; color: var(--muted); margin-bottom: 4px; }
        .field input, .field select {
            width: 100%; padding: 8px 10px; border: 1px solid #cbd5e1; border-radius: 6px;
            font-size: 0.9rem; background: #fff;
        }
        .btn-submit {
            margin-top: 18px; width: 100%; padding: 12px; background: var(--primary-light);
            color: white; border: none; border-radius: 8px; font-size: 1rem; font-weight: 700;
            cursor: pointer; transition: background 0.2s;
        }
        .btn-submit:hover { background: var(--primary); }
        .result-box {
            padding: 18px; border-radius: 8px; border-left: 6px solid var(--primary-light);
            background: #f8fafc; margin-bottom: 16px;
        }
        .badge {
            display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700;
            font-size: 0.85rem; color: white;
        }
        .badge-high { background: var(--danger); }
        .badge-med { background: var(--warning); }
        .badge-low { background: var(--success); }
        table { width: 100%; border-collapse: collapse; font-size: 0.84rem; margin-top: 8px; }
        th, td { border-bottom: 1px solid var(--border); padding: 8px 6px; text-align: left; }
        th { background: #f1f5f9; font-weight: 700; color: #334155; }
        .factor-list { list-style: none; margin-top: 8px; }
        .factor-list li {
            padding: 7px 10px; margin-bottom: 6px; border-radius: 6px; font-size: 0.86rem;
            background: #f1f5f9; border-left: 3px solid #64748b;
        }
    </style>
</head>
<body>
    <header>
        <h1>BÀI TẬP LỚN HỌC PHẦN NHẬP MÔN HỌC MÁY (FIT4013) — ĐỀ TÀI 13</h1>
        <p>Hệ thống Hỗ trợ Quyết định Nhân sự: Dự đoán Nhân viên có Khả năng Nghỉ việc (IBM HR Analytics)</p>
    </header>

    <div class="container">
        <div>
            <div class="card">
                <h2>1. Nhập Hồ sơ Nhân viên hoặc Chọn Mẫu Thử Nghiệm</h2>
                <div class="preset-bar">
                    <button type="button" class="btn-preset" onclick="loadPreset('high_risk')">Mẫu 1: Nhân viên Trẻ / OverTime / Lương thấp (Nguy cơ cao)</button>
                    <button type="button" class="btn-preset" onclick="loadPreset('mid_risk')">Mẫu 2: Nhân viên Tầm trung / Chưa thăng tiến (Nguy cơ vừa)</button>
                    <button type="button" class="btn-preset" onclick="loadPreset('low_risk')">Mẫu 3: Quản lý Cấp cao / Gắn bó lâu năm (Nguy cơ thấp)</button>
                </div>

                <form id="predictForm">
                    <div class="field" style="margin-bottom: 14px;">
                        <label>Lựa chọn Mô hình Học máy (Đã huấn luyện trên Luồng B - Cải tiến)</label>
                        <select id="model_key" name="model_key">
                            {% for k, (label, _) in models.items() %}
                            <option value="{{ k }}">{{ label }}</option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="form-grid">
                        <div class="field"><label>Tuổi (Age)</label><input type="number" name="Age" id="Age" value="26" min="18" max="65"></div>
                        <div class="field"><label>Thu nhập tháng (USD)</label><input type="number" name="MonthlyIncome" id="MonthlyIncome" value="2400" min="1000" max="20000"></div>
                        <div class="field"><label>Làm thêm giờ (OverTime)</label>
                            <select name="OverTime" id="OverTime"><option value="Yes" selected>Yes (Có)</option><option value="No">No (Không)</option></select>
                        </div>
                        <div class="field"><label>Tình trạng hôn nhân</label>
                            <select name="MaritalStatus" id="MaritalStatus"><option value="Single" selected>Single (Độc thân)</option><option value="Married">Married (Đã kết hôn)</option><option value="Divorced">Divorced (Ly hôn)</option></select>
                        </div>
                        <div class="field"><label>Vị trí công việc (JobRole)</label>
                            <select name="JobRole" id="JobRole">
                                <option value="Sales Representative" selected>Sales Representative</option>
                                <option value="Laboratory Technician">Laboratory Technician</option>
                                <option value="Research Scientist">Research Scientist</option>
                                <option value="Sales Executive">Sales Executive</option>
                                <option value="Human Resources">Human Resources</option>
                                <option value="Manufacturing Director">Manufacturing Director</option>
                                <option value="Healthcare Representative">Healthcare Representative</option>
                                <option value="Manager">Manager</option>
                                <option value="Research Director">Research Director</option>
                            </select>
                        </div>
                        <div class="field"><label>Phòng ban (Department)</label>
                            <select name="Department" id="Department">
                                <option value="Sales" selected>Sales</option>
                                <option value="Research & Development">Research & Development</option>
                                <option value="Human Resources">Human Resources</option>
                            </select>
                        </div>
                        <div class="field"><label>Tần suất công tác</label>
                            <select name="BusinessTravel" id="BusinessTravel">
                                <option value="Travel_Frequently" selected>Travel_Frequently</option>
                                <option value="Travel_Rarely">Travel_Rarely</option>
                                <option value="Non-Travel">Non-Travel</option>
                            </select>
                        </div>
                        <div class="field"><label>Khoảng cách đi làm (dặm)</label><input type="number" name="DistanceFromHome" id="DistanceFromHome" value="18" min="1" max="30"></div>
                        <div class="field"><label>Tổng năm kinh nghiệm</label><input type="number" name="TotalWorkingYears" id="TotalWorkingYears" value="2" min="0" max="40"></div>
                        <div class="field"><label>Số năm tại công ty</label><input type="number" name="YearsAtCompany" id="YearsAtCompany" value="1" min="0" max="40"></div>
                        <div class="field"><label>Số năm ở vị trí hiện tại</label><input type="number" name="YearsInCurrentRole" id="YearsInCurrentRole" value="1" min="0" max="20"></div>
                        <div class="field"><label>Số năm chưa thăng chức</label><input type="number" name="YearsSinceLastPromotion" id="YearsSinceLastPromotion" value="1" min="0" max="20"></div>
                        <div class="field"><label>Số năm với quản lý hiện tại</label><input type="number" name="YearsWithCurrManager" id="YearsWithCurrManager" value="1" min="0" max="20"></div>
                        <div class="field"><label>Số công ty từng làm</label><input type="number" name="NumCompaniesWorked" id="NumCompaniesWorked" value="3" min="0" max="10"></div>
                        <div class="field"><label>Mức hài lòng công việc (1-4)</label><input type="number" name="JobSatisfaction" id="JobSatisfaction" value="1" min="1" max="4"></div>
                        <div class="field"><label>Hài lòng môi trường (1-4)</label><input type="number" name="EnvironmentSatisfaction" id="EnvironmentSatisfaction" value="2" min="1" max="4"></div>
                        <div class="field"><label>Cân bằng công việc (1-4)</label><input type="number" name="WorkLifeBalance" id="WorkLifeBalance" value="2" min="1" max="4"></div>
                        <div class="field"><label>Quyền mua cổ phiếu (0-3)</label><input type="number" name="StockOptionLevel" id="StockOptionLevel" value="0" min="0" max="3"></div>
                    </div>

                    <button type="submit" class="btn-submit">Phân tích & Dự đoán Nguy cơ Nghỉ việc</button>
                </form>
            </div>
        </div>

        <div>
            <div class="card">
                <h2>2. Kết quả Dự đoán & Giải thích Đặc trưng (Luồng B)</h2>
                <div id="resultContainer">
                    <p style="color: var(--muted); font-size: 0.92rem;">Nhấn nút <b>"Phân tích & Dự đoán Nguy cơ Nghỉ việc"</b> để xem xác suất dự đoán từ tất cả các mô hình và các chỉ số kỹ thuật đặc trưng (Feature Engineering).</p>
                </div>
            </div>

            <div class="card">
                <h2>3. Bảng Tổng hợp Thực nghiệm (Test Set, N = 294)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Luồng</th>
                            <th>Mô hình</th>
                            <th>Acc</th>
                            <th>Recall</th>
                            <th>F1</th>
                            <th>AUC</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for r in summary %}
                        <tr {% if r.Test_F1 > 0.55 %}style="background:#eff6ff; font-weight:600;"{% endif %}>
                            <td>{{ r.Stream.split(' ')[0] }} {{ r.Stream.split(' ')[1] }}</td>
                            <td>{{ r.Model }}</td>
                            <td>{{ "%.2f"|format(r.Test_Accuracy * 100) }}%</td>
                            <td>{{ "%.2f"|format(r.Test_Recall * 100) }}%</td>
                            <td><b>{{ "%.4f"|format(r.Test_F1) }}</b></td>
                            <td>{{ "%.4f"|format(r.Test_ROC_AUC) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const PRESETS = {
            high_risk: {
                Age: 24, MonthlyIncome: 2100, OverTime: "Yes", MaritalStatus: "Single",
                JobRole: "Sales Representative", Department: "Sales", BusinessTravel: "Travel_Frequently",
                DistanceFromHome: 22, TotalWorkingYears: 2, YearsAtCompany: 1, YearsInCurrentRole: 0,
                YearsSinceLastPromotion: 0, YearsWithCurrManager: 0, NumCompaniesWorked: 3,
                JobSatisfaction: 1, EnvironmentSatisfaction: 1, WorkLifeBalance: 2, StockOptionLevel: 0
            },
            mid_risk: {
                Age: 33, MonthlyIncome: 4800, OverTime: "Yes", MaritalStatus: "Married",
                JobRole: "Laboratory Technician", Department: "Research & Development", BusinessTravel: "Travel_Rarely",
                DistanceFromHome: 12, TotalWorkingYears: 8, YearsAtCompany: 6, YearsInCurrentRole: 5,
                YearsSinceLastPromotion: 5, YearsWithCurrManager: 4, NumCompaniesWorked: 2,
                JobSatisfaction: 2, EnvironmentSatisfaction: 2, WorkLifeBalance: 2, StockOptionLevel: 1
            },
            low_risk: {
                Age: 46, MonthlyIncome: 13500, OverTime: "No", MaritalStatus: "Married",
                JobRole: "Research Director", Department: "Research & Development", BusinessTravel: "Travel_Rarely",
                DistanceFromHome: 4, TotalWorkingYears: 22, YearsAtCompany: 15, YearsInCurrentRole: 9,
                YearsSinceLastPromotion: 2, YearsWithCurrManager: 8, NumCompaniesWorked: 1,
                JobSatisfaction: 4, EnvironmentSatisfaction: 4, WorkLifeBalance: 3, StockOptionLevel: 2
            }
        };

        function loadPreset(name) {
            const p = PRESETS[name];
            for (const k in p) {
                const el = document.getElementById(k);
                if (el) el.value = p[k];
            }
            document.getElementById('predictForm').dispatchEvent(new Event('submit'));
        }

        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const payload = Object.fromEntries(formData.entries());
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            const badgeClass = data.prob >= 0.5 ? 'badge-high' : (data.prob >= 0.3 ? 'badge-med' : 'badge-low');
            let allModelsHtml = data.all_models.map(m =>
                `<tr><td>${m.name}</td><td><b>${(m.prob*100).toFixed(1)}%</b></td><td>${m.pred_label}</td></tr>`
            ).join('');
            let factorsHtml = data.engineered.map(f =>
                `<li><b>${f.name}</b>: ${f.value} — <i>${f.note}</i></li>`
            ).join('');

            document.getElementById('resultContainer').innerHTML = `
                <div class="result-box">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:1.05rem; font-weight:700;">Dự đoán: ${data.label}</span>
                        <span class="badge ${badgeClass}">Xác suất nghỉ việc: ${(data.prob * 100).toFixed(1)}%</span>
                    </div>
                    <p style="font-size:0.88rem; color:#334155;"><b>Khuyến nghị HR:</b> ${data.recommendation}</p>
                </div>
                <h3 style="font-size:0.92rem; margin: 10px 0 6px;">So sánh Dự đoán giữa các Mô hình (Luồng B):</h3>
                <table>
                    <thead><tr><th>Mô hình</th><th>Xác suất Yes</th><th>Kết luận</th></tr></thead>
                    <tbody>${allModelsHtml}</tbody>
                </table>
                <h3 style="font-size:0.92rem; margin: 14px 0 6px;">Các Đặc trưng Cải tiến (Stream B Engineered Features):</h3>
                <ul class="factor-list">${factorsHtml}</ul>
            `;
        });

        window.addEventListener('DOMContentLoaded', () => {
            document.getElementById('predictForm').dispatchEvent(new Event('submit'));
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, models=MODELS, summary=EXPERIMENT_SUMMARY)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    model_key = data.get("model_key", "gnb")

    # Construct full 34-feature row matching training schema
    row = {
        "Age": int(data.get("Age", 30)),
        "BusinessTravel": data.get("BusinessTravel", "Travel_Rarely"),
        "DailyRate": 800,
        "Department": data.get("Department", "Research & Development"),
        "DistanceFromHome": int(data.get("DistanceFromHome", 5)),
        "Education": 3,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 9999,
        "EnvironmentSatisfaction": int(data.get("EnvironmentSatisfaction", 3)),
        "Gender": "Male",
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": data.get("JobRole", "Sales Executive"),
        "JobSatisfaction": int(data.get("JobSatisfaction", 3)),
        "MaritalStatus": data.get("MaritalStatus", "Single"),
        "MonthlyIncome": int(data.get("MonthlyIncome", 4000)),
        "MonthlyRate": 14000,
        "NumCompaniesWorked": int(data.get("NumCompaniesWorked", 1)),
        "Over18": "Y",
        "OverTime": data.get("OverTime", "No"),
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StandardHours": 80,
        "StockOptionLevel": int(data.get("StockOptionLevel", 0)),
        "TotalWorkingYears": int(data.get("TotalWorkingYears", 5)),
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": int(data.get("WorkLifeBalance", 3)),
        "YearsAtCompany": int(data.get("YearsAtCompany", 3)),
        "YearsInCurrentRole": int(data.get("YearsInCurrentRole", 2)),
        "YearsSinceLastPromotion": int(data.get("YearsSinceLastPromotion", 1)),
        "YearsWithCurrManager": int(data.get("YearsWithCurrManager", 2)),
    }
    X_input = pd.DataFrame([row])

    # Compute engineered features for display
    fe_df = HRFeatureEngineer(drop_noisy=True).fit_transform(X_input)
    fe_row = fe_df.iloc[0]

    selected_label, selected_pipe = MODELS.get(model_key, MODELS["gnb"])
    prob = float(selected_pipe.predict_proba(X_input)[0, 1])
    pred = int(selected_pipe.predict(X_input)[0])

    all_models_res = []
    for k, (m_label, m_pipe) in MODELS.items():
        p = float(m_pipe.predict_proba(X_input)[0, 1])
        pr = int(m_pipe.predict(X_input)[0])
        all_models_res.append({
            "name": m_label.split(" [")[0],
            "prob": round(p, 4),
            "pred_label": "Nghỉ việc (Yes)" if pr == 1 else "Ở lại (No)"
        })

    if prob >= 0.5:
        rec = "Nhân sự thuộc nhóm NGUY CƠ NGHỈ VIỆC CAO. Cần rà soát ngay khối lượng làm thêm giờ (OverTime), chế độ đãi ngộ hoặc tổ chức buổi trao đổi 1-1 (Stay Interview)."
    elif prob >= 0.3:
        rec = "Nhân sự có dấu hiệu rủi ro trung bình. Nên theo dõi mức độ hài lòng môi trường làm việc và lộ trình thăng tiến trong kỳ đánh giá tới."
    else:
        rec = "Nhân sự có mức độ gắn bó ổn định. Duy trì chính sách phúc lợi và lộ trình phát triển hiện tại."

    engineered_info = [
        {"name": "TotalSatisfaction", "value": f"{fe_row['TotalSatisfaction']:.2f} / 4.00", "note": "Điểm hài lòng tổng hợp (Môi trường, Công việc, Quan hệ, Work-Life)"},
        {"name": "IncomePerWorkingYear", "value": f"{fe_row['IncomePerWorkingYear']:.1f} USD/năm", "note": "Thu nhập bình quân trên mỗi năm kinh nghiệm"},
        {"name": "StagnationRatio", "value": f"{fe_row['StagnationRatio']:.2f}", "note": "Tỷ lệ số năm chưa được thăng chức / Thâm niên công ty"},
        {"name": "CommuteBurden", "value": f"{fe_row['CommuteBurden']:.0f}", "note": "Áp lực khoảng cách đi làm kết hợp làm thêm giờ"},
        {"name": "NewHireRisk", "value": "Có (1)" if fe_row['NewHireRisk'] == 1 else "Không (0)", "note": "Rủi ro nhân sự mới gia nhập (<= 2 năm tại công ty hoặc <= 1 năm ở vị trí mới)"}
    ]

    return jsonify({
        "model": selected_label,
        "prob": round(prob, 4),
        "pred": pred,
        "label": "CÓ NGUY CƠ NGHỈ VIỆC (Yes)" if pred == 1 else "ỔN ĐỊNH / Ở LẠI (No)",
        "recommendation": rec,
        "all_models": all_models_res,
        "engineered": engineered_info
    })


if __name__ == "__main__":
    print("Starting HR Attrition Demo Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)

