"""
HR Employee Attrition Prediction Package (FIT4013 - Topic 13)
"""
from .pipeline import (
    HRFeatureEngineer,
    load_and_split_data,
    build_preprocessor_stream_a,
    build_preprocessor_stream_b,
    RANDOM_STATE,
    DROP_COLS_BASIC,
    CAT_COLS
)

