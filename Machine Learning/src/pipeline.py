import os
import json
import joblib
import numpy as np
import pandas as pd

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.base import BaseEstimator, TransformerMixin

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set visual style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
sns.set_theme(style="whitegrid", palette="muted")

DROP_COLS_BASIC = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
CAT_COLS = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']


class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for Stream B (Improved Pipeline).
    Creates domain-informed HR features and removes noisy/collinear columns.
    """
    def __init__(self, drop_noisy=True):
        self.drop_noisy = drop_noisy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # 1. Total Satisfaction Index (mean of 4 survey scores)
        X['TotalSatisfaction'] = (
            X['EnvironmentSatisfaction'] +
            X['JobSatisfaction'] +
            X['RelationshipSatisfaction'] +
            X['WorkLifeBalance']
        ) / 4.0

        # 2. Income Stability / Compensation relative to experience
        X['IncomePerWorkingYear'] = X['MonthlyIncome'] / (X['TotalWorkingYears'] + 1.0)

        # 3. Career Stagnation Ratio
        X['StagnationRatio'] = X['YearsSinceLastPromotion'] / (X['YearsAtCompany'] + 1.0)

        # 4. Job Hopping Rate
        X['JobHoppingRate'] = X['NumCompaniesWorked'] / (X['TotalWorkingYears'] + 1.0)

        # 5. Early Tenure Risk (first 2 years at company or <=1 year in role)
        X['NewHireRisk'] = ((X['YearsAtCompany'] <= 2) | (X['YearsInCurrentRole'] <= 1)).astype(int)

        # 6. Interaction risk factors: OverTime with MaritalStatus, Satisfaction, and Distance
        ot_bin = (X['OverTime'] == 'Yes').astype(int)
        single_bin = (X['MaritalStatus'] == 'Single').astype(int)
        X['OverTime_Single_Risk'] = ot_bin * single_bin
        X['OverTime_LowSat_Risk'] = ot_bin * (X['TotalSatisfaction'] <= 2.5).astype(int)
        X['CommuteBurden'] = X['DistanceFromHome'] * (1 + ot_bin)

        cols_to_drop = list(DROP_COLS_BASIC)
        if self.drop_noisy:
            # Drop synthetic uniform rate columns (~0 correlation) and multicollinear JobLevel (r=0.95 with MonthlyIncome)
            cols_to_drop += ['DailyRate', 'HourlyRate', 'MonthlyRate', 'JobLevel', 'PercentSalaryHike']
        X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
        return X


def load_and_split_data(csv_path, test_size=0.2, random_state=RANDOM_STATE):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['Attrition'])
    y = (df['Attrition'] == 'Yes').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return df, X_train, X_test, y_train, y_test


def build_preprocessor_stream_a(X_sample):
    num_cols = [c for c in X_sample.columns if c not in DROP_COLS_BASIC + CAT_COLS]
    return ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CAT_COLS)
    ], remainder='drop')


def build_preprocessor_stream_b(X_sample, k_best=20):
    fe_sample = HRFeatureEngineer(drop_noisy=True).fit_transform(X_sample)
    cat_cols_b = [c for c in CAT_COLS if c in fe_sample.columns]
    num_cols_b = [c for c in fe_sample.columns if c not in cat_cols_b]
    steps = [
        ('fe', HRFeatureEngineer(drop_noisy=True)),
        ('ct', ColumnTransformer([
            ('num', RobustScaler(), num_cols_b),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols_b)
        ], remainder='drop'))
    ]
    if k_best is not None:
        steps.append(('sel', SelectKBest(score_func=f_classif, k=k_best)))
    return Pipeline(steps)
