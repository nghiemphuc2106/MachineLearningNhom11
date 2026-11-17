import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", ".mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from src.pipeline import (
    HRFeatureEngineer,
    load_and_split_data,
    build_preprocessor_stream_a,
    build_preprocessor_stream_b,
    RANDOM_STATE,
    DROP_COLS_BASIC,
    CAT_COLS
)

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
RES_DIR = os.path.join(BASE_DIR, "outputs", "results")
MOD_DIR = os.path.join(BASE_DIR, "outputs", "models")

for d in [FIG_DIR, RES_DIR, MOD_DIR]:
    os.makedirs(d, exist_ok=True)


def generate_eda_figures(df):
    print("1. Generating EDA figures...")
    df_eda = df.copy()
    df_eda['Attrition_num'] = (df_eda['Attrition'] == 'Yes').astype(int)

    # Fig 1: Target distribution
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    counts = df_eda['Attrition'].value_counts()
    colors = ['#2b5c8f', '#d95f02']
    bars = ax.bar(['No (Ở lại)', 'Yes (Nghỉ việc)'], [counts['No'], counts['Yes']], color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        pct = h / len(df_eda) * 100
        ax.text(bar.get_x() + bar.get_width()/2, h + 25, f"{h} ({pct:.1f}%)", ha='center', va='bottom', fontweight='bold', fontsize=10.5)
    ax.set_ylim(0, 1400)
    ax.set_title("Phân phối biến mục tiêu Attrition (N = 1,470)", fontsize=12.5, fontweight='bold', pad=12)
    ax.set_ylabel("Số lượng nhân viên", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_target_distribution.png"))
    plt.close()

    # Fig 2: Numerical distributions & Boxplots by Attrition
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200)
    num_features = [
        ('Age', 'Tuổi (Age)'),
        ('MonthlyIncome', 'Thu nhập hàng tháng (MonthlyIncome - USD)'),
        ('TotalWorkingYears', 'Tổng số năm kinh nghiệm (TotalWorkingYears)'),
        ('DistanceFromHome', 'Khoảng cách từ nhà đến công ty (DistanceFromHome - dặm)')
    ]
    for idx, (col, title) in enumerate(num_features):
        r, c = idx // 2, idx % 2
        sns.boxplot(data=df_eda, x='Attrition', y=col, hue='Attrition', legend=False, ax=axes[r, c], palette={'No': '#4c72b0', 'Yes': '#dd8452'}, width=0.45)
        axes[r, c].set_title(title, fontsize=11, fontweight='bold')
        axes[r, c].set_xlabel("Nghỉ việc (Attrition)")
        axes[r, c].set_ylabel(col)
    plt.suptitle("So sánh phân phối và ngoại lệ (Outliers) của các đặc trưng số theo nhóm Attrition", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_numerical_boxplots.png"), bbox_inches='tight')
    plt.close()

    # Fig 3: Categorical attrition rates (OverTime, JobRole, BusinessTravel, MaritalStatus)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=200)
    cat_plot_cols = [
        ('OverTime', 'Tỷ lệ nghỉ việc theo Làm thêm giờ (OverTime)'),
        ('MaritalStatus', 'Tỷ lệ nghỉ việc theo Tình trạng hôn nhân'),
        ('BusinessTravel', 'Tỷ lệ nghỉ việc theo Tần suất công tác'),
        ('JobRole', 'Tỷ lệ nghỉ việc theo Vị trí công việc (JobRole)')
    ]
    for idx, (col, title) in enumerate(cat_plot_cols):
        r, c = idx // 2, idx % 2
        rate_df = df_eda.groupby(col)['Attrition_num'].agg(['count', 'mean']).reset_index()
        rate_df['mean_pct'] = rate_df['mean'] * 100
        rate_df = rate_df.sort_values(by='mean_pct', ascending=True)
        axes[r, c].barh(rate_df[col], rate_df['mean_pct'], color='#3b75af', edgecolor='black', height=0.6)
        for i, (v, cnt) in enumerate(zip(rate_df['mean_pct'], rate_df['count'])):
            axes[r, c].text(v + 0.8, i, f"{v:.1f}% (n={cnt})", va='center', fontsize=9)
        axes[r, c].set_xlim(0, max(rate_df['mean_pct']) + 12)
        axes[r, c].set_title(title, fontsize=11, fontweight='bold')
        axes[r, c].set_xlabel("Tỷ lệ nghỉ việc (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_categorical_rates.png"), bbox_inches='tight')
    plt.close()

    # Fig 4: Correlation Heatmap of key numerical variables
    key_num = [
        'Age', 'MonthlyIncome', 'JobLevel', 'TotalWorkingYears',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager', 'NumCompaniesWorked', 'DistanceFromHome',
        'PercentSalaryHike', 'PerformanceRating', 'DailyRate', 'Attrition_num'
    ]
    corr_mat = df_eda[key_num].corr()
    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=200)
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.3, vmax=1.0,
                cbar_kws={'shrink': 0.8}, linewidths=0.5, ax=ax, annot_kws={'size': 8.5})
    ax.set_title("Ma trận tương quan Pearson giữa các đặc trưng số chính và Attrition", fontsize=12.5, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_correlation_heatmap.png"), bbox_inches='tight')
    plt.close()


def run_all_experiments(X_train, X_test, y_train, y_test):
    print("2. Running experiments across Stream A and Stream B...")
    prep_A = build_preprocessor_stream_a(X_train)
    prep_B_all = build_preprocessor_stream_b(X_train, k_best=None)
    prep_B_k15 = build_preprocessor_stream_b(X_train, k_best=15)

    experiment_configs = [
        # Stream A: Basic Pipeline
        ("Luồng A (Cơ bản)", "K-NN (Baseline, k=5, Euclidean)", prep_A,
         KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')),
        ("Luồng A (Cơ bản)", "K-NN (Tuned, k=11, Manhattan, dist)", prep_A,
         KNeighborsClassifier(n_neighbors=11, weights='distance', metric='manhattan')),
        ("Luồng A (Cơ bản)", "Gaussian NB (Baseline, smooth=1e-9)", prep_A,
         GaussianNB(var_smoothing=1e-9)),
        ("Luồng A (Cơ bản)", "Gaussian NB (Tuned, smooth=0.1)", prep_A,
         GaussianNB(var_smoothing=0.1)),
        ("Luồng A (Cơ bản)", "Decision Tree (Baseline, mặc định)", prep_A,
         DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ("Luồng A (Cơ bản)", "Decision Tree (Tuned, depth=4, balanced)", prep_A,
         DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, class_weight='balanced', random_state=RANDOM_STATE)),

        # Stream B: Improved Pipeline (Feature Engineering + RobustScaler + Noise/Collinearity Removal)
        ("Luồng B (Cải tiến)", "K-NN (Baseline, k=5, Euclidean)", prep_B_k15,
         KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')),
        ("Luồng B (Cải tiến)", "K-NN (Tuned, k=7, Manhattan, dist)", prep_B_k15,
         KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan')),
        ("Luồng B (Cải tiến)", "Gaussian NB (Baseline, smooth=1e-9)", prep_B_k15,
         GaussianNB(var_smoothing=1e-9)),
        ("Luồng B (Cải tiến)", "Gaussian NB (Tuned, smooth=0.1)", prep_B_all,
         GaussianNB(var_smoothing=0.1)),
        ("Luồng B (Cải tiến)", "Decision Tree (Baseline, mặc định)", prep_B_all,
         DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ("Luồng B (Cải tiến)", "Decision Tree (Tuned, depth=4, balanced)", prep_B_all,
         DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, class_weight='balanced', random_state=RANDOM_STATE)),

        # Bonus Extension: Ensemble Random Forest
        ("Mở rộng (Ensemble)", "Random Forest (Tuned, 200 cây, balanced)", prep_B_k15,
         RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=5, class_weight='balanced', random_state=RANDOM_STATE))
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    trained_pipelines = {}

    for stream_name, model_name, prep, clf in experiment_configs:
        pipe = Pipeline([('prep', prep), ('clf', clf)])
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            n_jobs=1
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        res_row = {
            'Stream': stream_name,
            'Model': model_name,
            'CV_F1_mean': round(float(np.mean(cv_scores['test_f1'])), 4),
            'CV_F1_std': round(float(np.std(cv_scores['test_f1'])), 4),
            'CV_AUC_mean': round(float(np.mean(cv_scores['test_roc_auc'])), 4),
            'Test_Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'Test_Precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'Test_Recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            'Test_F1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            'Test_ROC_AUC': round(float(roc_auc_score(y_test, y_prob)), 4),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
        }
        results.append(res_row)
        key = f"{stream_name} | {model_name}"
        trained_pipelines[key] = (pipe, y_pred, y_prob)
        print(f"  {key:58s} -> F1={res_row['Test_F1']:.4f}, Rec={res_row['Test_Recall']:.4f}, AUC={res_row['Test_ROC_AUC']:.4f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RES_DIR, "experiment_summary.csv"), index=False, encoding='utf-8-sig')
    with open(os.path.join(RES_DIR, "experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save best models for Web Demo
    best_gnb_pipe = trained_pipelines["Luồng B (Cải tiến) | Gaussian NB (Tuned, smooth=0.1)"][0]
    best_dt_pipe = trained_pipelines["Luồng B (Cải tiến) | Decision Tree (Tuned, depth=4, balanced)"][0]
    best_knn_pipe = trained_pipelines["Luồng B (Cải tiến) | K-NN (Tuned, k=7, Manhattan, dist)"][0]
    best_rf_pipe = trained_pipelines["Mở rộng (Ensemble) | Random Forest (Tuned, 200 cây, balanced)"][0]

    joblib.dump(best_gnb_pipe, os.path.join(MOD_DIR, "best_gnb_stream_b.pkl"))
    joblib.dump(best_dt_pipe, os.path.join(MOD_DIR, "best_dt_stream_b.pkl"))
    joblib.dump(best_knn_pipe, os.path.join(MOD_DIR, "best_knn_stream_b.pkl"))
    joblib.dump(best_rf_pipe, os.path.join(MOD_DIR, "best_rf_extension.pkl"))

    return res_df, trained_pipelines


def generate_evaluation_figures(X_train, X_test, y_train, y_test, res_df, trained_pipelines):
    print("3. Generating evaluation and error analysis figures...")

    # Fig 5: Hyperparameter Sensitivity Curves (K-NN k, Naive Bayes var_smoothing, Decision Tree max_depth)
    prep_B_k15 = build_preprocessor_stream_b(X_train, k_best=15)
    prep_B_all = build_preprocessor_stream_b(X_train, k_best=None)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), dpi=200)

    # 5a: K-NN k sensitivity
    k_vals = [3, 5, 7, 9, 11, 15, 19, 25]
    knn_f1_u, knn_f1_d = [], []
    for k in k_vals:
        p1 = Pipeline([('p', prep_B_k15), ('c', KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='manhattan'))]).fit(X_train, y_train)
        p2 = Pipeline([('p', prep_B_k15), ('c', KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan'))]).fit(X_train, y_train)
        knn_f1_u.append(f1_score(y_test, p1.predict(X_test)))
        knn_f1_d.append(f1_score(y_test, p2.predict(X_test)))
    axes[0].plot(k_vals, knn_f1_u, 'o--', label='Uniform weights', color='#3b75af')
    axes[0].plot(k_vals, knn_f1_d, 's-', label='Distance weights', color='#d95f02')
    axes[0].set_title("K-NN: Ảnh hưởng của số láng giềng k", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Số láng giềng (k)")
    axes[0].set_ylabel("Test F1-score (Lớp Yes)")
    axes[0].legend()

    # 5b: GNB var_smoothing sensitivity
    smooth_vals = np.logspace(-9, 0, 10)
    gnb_f1, gnb_rec, gnb_prec = [], [], []
    for s in smooth_vals:
        p = Pipeline([('p', prep_B_all), ('c', GaussianNB(var_smoothing=s))]).fit(X_train, y_train)
        pred = p.predict(X_test)
        gnb_f1.append(f1_score(y_test, pred, zero_division=0))
        gnb_rec.append(recall_score(y_test, pred, zero_division=0))
        gnb_prec.append(precision_score(y_test, pred, zero_division=0))
    axes[1].semilogx(smooth_vals, gnb_f1, 'o-', label='F1-score', color='#2ca02c', linewidth=2)
    axes[1].semilogx(smooth_vals, gnb_rec, 's--', label='Recall', color='#d95f02')
    axes[1].semilogx(smooth_vals, gnb_prec, '^:', label='Precision', color='#3b75af')
    axes[1].set_title("Gaussian NB: Ảnh hưởng của var_smoothing", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("var_smoothing (log scale)")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    # 5c: Decision Tree max_depth sensitivity
    depths = [2, 3, 4, 5, 6, 8, 10, 15]
    dt_train_f1, dt_test_f1 = [], []
    for d in depths:
        p = Pipeline([('p', prep_B_all), ('c', DecisionTreeClassifier(max_depth=d, min_samples_leaf=15, class_weight='balanced', random_state=RANDOM_STATE))]).fit(X_train, y_train)
        dt_train_f1.append(f1_score(y_train, p.predict(X_train)))
        dt_test_f1.append(f1_score(y_test, p.predict(X_test)))
    axes[2].plot(depths, dt_train_f1, 'o--', label='Train F1-score', color='#7f7f7f')
    axes[2].plot(depths, dt_test_f1, 's-', label='Test F1-score', color='#d62728', linewidth=2)
    axes[2].set_title("Decision Tree: Ảnh hưởng của max_depth", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Độ sâu tối đa (max_depth)")
    axes[2].set_ylabel("F1-score (Lớp Yes)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_hyperparameter_tuning.png"), bbox_inches='tight')
    plt.close()

    # Fig 6: Stream A vs Stream B Comparison Bar Chart (F1-score and Recall across tuned models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
    models_compare = ["K-NN (Tuned)", "Gaussian NB (Tuned)", "Decision Tree (Tuned)"]
    f1_A = [
        res_df.loc[(res_df['Stream'] == 'Luồng A (Cơ bản)') & (res_df['Model'].str.contains('K-NN \\(Tuned')), 'Test_F1'].values[0],
        res_df.loc[(res_df['Stream'] == 'Luồng A (Cơ bản)') & (res_df['Model'].str.contains('Gaussian NB \\(Tuned')), 'Test_F1'].values[0],
        res_df.loc[(res_df['Stream'] == 'Luồng A (Cơ bản)') & (res_df['Model'].str.contains('Decision Tree \\(Tuned')), 'Test_F1'].values[0],
    ]
    f1_B = [
        res_df.loc[(res_df['Stream'] == 'Luồng B (Cải tiến)') & (res_df['Model'].str.contains('K-NN \\(Tuned')), 'Test_F1'].values[0],
        res_df.loc[(res_df['Stream'] == 'Luồng B (Cải tiến)') & (res_df['Model'].str.contains('Gaussian NB \\(Tuned')), 'Test_F1'].values[0],
        res_df.loc[(res_df['Stream'] == 'Luồng B (Cải tiến)') & (res_df['Model'].str.contains('Decision Tree \\(Tuned')), 'Test_F1'].values[0],
    ]
    auc_A = [
        res_df.loc[(res_df['Stream'] == 'Luồng A (Cơ bản)') & (res_df['Model'].str.contains('K-NN \\(Tuned')), 'Test_ROC_AUC'].values[0],
        res_df.loc[(res_df['Stream'] == 'Luồng A (Cơ bản)') & (res_df['Model'].str.contains('Gaussian NB \\(Tuned')), 'Test_ROC_AUC'].values[0],
        res_df.loc[(res_df['Stream'] == 'Luồng A (Cơ bản)') & (res_df['Model'].str.contains('Decision Tree \\(Tuned')), 'Test_ROC_AUC'].values[0],
    ]
    auc_B = [
        res_df.loc[(res_df['Stream'] == 'Luồng B (Cải tiến)') & (res_df['Model'].str.contains('K-NN \\(Tuned')), 'Test_ROC_AUC'].values[0],
        res_df.loc[(res_df['Stream'] == 'Luồng B (Cải tiến)') & (res_df['Model'].str.contains('Gaussian NB \\(Tuned')), 'Test_ROC_AUC'].values[0],
        res_df.loc[(res_df['Stream'] == 'Luồng B (Cải tiến)') & (res_df['Model'].str.contains('Decision Tree \\(Tuned')), 'Test_ROC_AUC'].values[0],
    ]

    x = np.arange(len(models_compare))
    w = 0.35
    b1 = axes[0].bar(x - w/2, f1_A, w, label='Luồng A (Pipeline cơ bản)', color='#6baed6', edgecolor='black')
    b2 = axes[0].bar(x + w/2, f1_B, w, label='Luồng B (Pipeline cải tiến)', color='#fd8d3c', edgecolor='black')
    for b in list(b1) + list(b2):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha='center', fontsize=9.5, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models_compare)
    axes[0].set_ylim(0, 0.70)
    axes[0].set_ylabel("F1-score (Lớp nghỉ việc)")
    axes[0].set_title("So sánh F1-score giữa Luồng A và Luồng B", fontsize=11.5, fontweight='bold')
    axes[0].legend()

    b3 = axes[1].bar(x - w/2, auc_A, w, label='Luồng A (Pipeline cơ bản)', color='#6baed6', edgecolor='black')
    b4 = axes[1].bar(x + w/2, auc_B, w, label='Luồng B (Pipeline cải tiến)', color='#fd8d3c', edgecolor='black')
    for b in list(b3) + list(b4):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha='center', fontsize=9.5, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models_compare)
    axes[1].set_ylim(0, 0.90)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("So sánh ROC-AUC giữa Luồng A và Luồng B", fontsize=11.5, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_stream_comparison.png"), bbox_inches='tight')
    plt.close()

    # Fig 7: Confusion Matrices for key models in Stream B
    selected_keys = [
        ("Luồng B (Cải tiến) | K-NN (Tuned, k=7, Manhattan, dist)", "K-NN (Tuned - Luồng B)"),
        ("Luồng B (Cải tiến) | Gaussian NB (Tuned, smooth=0.1)", "Gaussian NB (Tuned - Luồng B)"),
        ("Luồng B (Cải tiến) | Decision Tree (Tuned, depth=4, balanced)", "Decision Tree (Tuned - Luồng B)"),
        ("Mở rộng (Ensemble) | Random Forest (Tuned, 200 cây, balanced)", "Random Forest (Mở rộng)")
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=200)
    for idx, (k, title) in enumerate(selected_keys):
        _, y_pred, _ = trained_pipelines[k]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=['Dự đoán: No', 'Dự đoán: Yes'],
                    yticklabels=['Thực tế: No', 'Thực tế: Yes'],
                    annot_kws={'size': 12, 'weight': 'bold'})
        axes[idx].set_title(title, fontsize=10.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_confusion_matrices.png"), bbox_inches='tight')
    plt.close()

    # Fig 8: ROC Curves comparison
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200)
    roc_items = [
        ("Luồng A (Cơ bản) | K-NN (Baseline, k=5, Euclidean)", "K-NN Baseline (Luồng A)", "--", "#7f7f7f"),
        ("Luồng B (Cải tiến) | K-NN (Tuned, k=7, Manhattan, dist)", "K-NN Tuned (Luồng B)", "-.", "#1f77b4"),
        ("Luồng B (Cải tiến) | Decision Tree (Tuned, depth=4, balanced)", "Decision Tree Tuned (Luồng B)", ":", "#ff7f0e"),
        ("Luồng B (Cải tiến) | Gaussian NB (Tuned, smooth=0.1)", "Gaussian NB Tuned (Luồng B)", "-", "#2ca02c"),
        ("Mở rộng (Ensemble) | Random Forest (Tuned, 200 cây, balanced)", "Random Forest (Mở rộng)", "-", "#d62728"),
    ]
    for key, label, ls, color in roc_items:
        _, _, y_prob = trained_pipelines[key]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, linestyle=ls, color=color, linewidth=2, label=f"{label} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Ngẫu nhiên (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("Đường cong ROC trên tập Kiểm thử (Test Set, N = 294)", fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_roc_curves.png"), bbox_inches='tight')
    plt.close()

    # Fig 9: Decision Tree Feature Importance & Top Engineered Features
    dt_pipe = trained_pipelines["Luồng B (Cải tiến) | Decision Tree (Tuned, depth=4, balanced)"][0]
    ct = dt_pipe.named_steps['prep'].named_steps['ct']
    fe_sample = HRFeatureEngineer(drop_noisy=True).fit_transform(X_train)
    cat_cols_b = [c for c in CAT_COLS if c in fe_sample.columns]
    num_cols_b = [c for c in fe_sample.columns if c not in cat_cols_b]
    ohe_names = list(ct.named_transformers_['cat'].get_feature_names_out(cat_cols_b))
    feature_names = num_cols_b + ohe_names
    importances = dt_pipe.named_steps['clf'].feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    imp_df_sorted = imp_df.sort_values(by='Importance', ascending=True)
    ax.barh(imp_df_sorted['Feature'], imp_df_sorted['Importance'], color='#2b5c8f', edgecolor='black')
    for i, v in enumerate(imp_df_sorted['Importance']):
        if v > 0:
            ax.text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=9.5)
    ax.set_xlim(0, max(imp_df_sorted['Importance']) + 0.08)
    ax.set_xlabel("Mức độ quan trọng (Gini Importance)")
    ax.set_title("Top các đặc trưng quan trọng nhất trong Cây quyết định (Luồng B)", fontsize=11.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig9_feature_importance.png"), bbox_inches='tight')
    plt.close()

    # Extract specific False Negatives and False Positives for Error Analysis in Report
    gnb_pipe, gnb_pred, gnb_prob = trained_pipelines["Luồng B (Cải tiến) | Gaussian NB (Tuned, smooth=0.1)"]
    err_df = X_test.copy()
    err_df['Actual'] = y_test.values
    err_df['Predicted'] = gnb_pred
    err_df['Prob_Yes'] = np.round(gnb_prob, 4)

    fn_cases = err_df[(err_df['Actual'] == 1) & (err_df['Predicted'] == 0)].sort_values(by='Prob_Yes', ascending=True)
    fp_cases = err_df[(err_df['Actual'] == 0) & (err_df['Predicted'] == 1)].sort_values(by='Prob_Yes', ascending=False)

    cols_inspect = ['EmployeeNumber', 'Age', 'JobRole', 'MonthlyIncome', 'OverTime', 'MaritalStatus',
                    'TotalWorkingYears', 'YearsAtCompany', 'JobSatisfaction', 'EnvironmentSatisfaction',
                    'DistanceFromHome', 'Actual', 'Predicted', 'Prob_Yes']
    fn_cases[cols_inspect].head(10).to_csv(os.path.join(RES_DIR, "error_analysis_FN.csv"), index=False)
    fp_cases[cols_inspect].head(10).to_csv(os.path.join(RES_DIR, "error_analysis_FP.csv"), index=False)
    print("4. Saved error analysis cases (FN and FP) to outputs/results/.")


if __name__ == "__main__":
    df, X_train, X_test, y_train, y_test = load_and_split_data(CSV_PATH)
    generate_eda_figures(df)
    res_df, trained_pipelines = run_all_experiments(X_train, X_test, y_train, y_test)
    generate_evaluation_figures(X_train, X_test, y_train, y_test, res_df, trained_pipelines)
    print("Done! All figures, models, and metrics have been generated.")
