# ==============================================================================
# IntelliDS AI
# Data Preparation Helper Functions
# ==============================================================================

import pandas as pd
import numpy as np
import re
import os

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_object_dtype,
)

# ==============================================================================
# Identifier Keywords
# ==============================================================================

IDENTIFIER_PATTERNS = [
    "id",
    "cust",
    "customer",
    "account",
    "acc",
    "loan",
    "mobile",
    "phone",
    "email",
    "pan",
    "aadhaar",
    "aadhar",
    "ifsc",
    "transaction",
    "txn",
    "uuid",
    "reference",
    "ref",
]


# ==============================================================================
# Safe Percentage
# ==============================================================================


def percentage(value, total):

    if total == 0:
        return 0

    return round((value / total) * 100, 2)


# ==============================================================================
# Memory Formatter
# ==============================================================================


def format_memory(bytes_used):

    return round(bytes_used / 1024 / 1024, 2)


# ==============================================================================
# Detect Possible Datetime Columns
# ==============================================================================


def detect_datetime_columns(df):

    detected = []

    for column in df.columns:

        if is_datetime64_any_dtype(df[column]):

            detected.append(column)

            continue

        if not is_object_dtype(df[column]):
            continue

        converted = pd.to_datetime(
            df[column],
            errors="coerce",
            infer_datetime_format=True,
        )

        success_rate = converted.notna().mean()

        if success_rate >= 0.80:
            detected.append(column)

    return detected


# ==============================================================================
# Detect Identifier Columns
# ==============================================================================


def detect_identifier_columns(df):

    identifiers = []

    for column in df.columns:

        lower = column.lower()

        unique_ratio = df[column].nunique(dropna=False) / max(len(df), 1)

        keyword_match = any(keyword in lower for keyword in IDENTIFIER_PATTERNS)

        if keyword_match:

            identifiers.append(column)

            continue

        if unique_ratio > 0.98:

            identifiers.append(column)

    return sorted(list(set(identifiers)))


# ==============================================================================
# Constant Columns
# ==============================================================================


def detect_constant_columns(df):

    constant = []

    for column in df.columns:

        if df[column].nunique(dropna=False) <= 1:

            constant.append(column)

    return constant


# ==============================================================================
# Feature Type Detection
# ==============================================================================


def detect_feature_types(df):

    numeric = []
    categorical = []
    boolean = []
    datetime = []
    text = []

    detected_datetime = detect_datetime_columns(df)

    for column in df.columns:

        if column in detected_datetime:

            datetime.append(column)

            continue

        if is_numeric_dtype(df[column]):

            numeric.append(column)

            continue

        if is_bool_dtype(df[column]):

            boolean.append(column)

            continue

        if is_object_dtype(df[column]):

            unique = df[column].nunique(dropna=True)

            if unique > 50:

                text.append(column)

            else:

                categorical.append(column)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "boolean": boolean,
        "datetime": datetime,
        "text": text,
    }


# ==============================================================================
# Memory Optimization Estimate
# ==============================================================================


def estimate_memory_saving(df):

    current = df.memory_usage(deep=True).sum()

    optimized = current

    for column in df.columns:

        if is_numeric_dtype(df[column]):

            optimized -= df[column].memory_usage(deep=True) * 0.35

        elif is_object_dtype(df[column]):

            ratio = df[column].nunique() / max(len(df), 1)

            if ratio < 0.40:

                optimized -= df[column].memory_usage(deep=True) * 0.60

    optimized = max(optimized, 0)

    saved = current - optimized

    return {
        "current": round(current / 1024 / 1024, 2),
        "optimized": round(optimized / 1024 / 1024, 2),
        "saved": round(saved / 1024 / 1024, 2),
        "saving_percent": percentage(saved, current),
    }


# ==============================================================================
# Dataset Health Score
# ==============================================================================


def calculate_health_score(df):

    score = 100

    duplicate_rows = int(df.duplicated().sum())

    duplicate_percent = percentage(
        duplicate_rows,
        len(df),
    )

    missing_percent = percentage(int(df.isna().sum().sum()), df.shape[0] * df.shape[1])

    constant_columns = len(detect_constant_columns(df))

    identifier_columns = len(detect_identifier_columns(df))

    score -= min(
        missing_percent * 0.35,
        30,
    )

    score -= min(
        duplicate_percent * 0.50,
        20,
    )

    score -= constant_columns * 2

    score -= identifier_columns * 1

    score = max(
        min(round(score), 100),
        0,
    )

    if score >= 90:

        grade = "A"

        status = "Excellent"

    elif score >= 80:

        grade = "B"

        status = "Good"

    elif score >= 70:

        grade = "C"

        status = "Fair"

    elif score >= 60:

        grade = "D"

        status = "Poor"

    else:

        grade = "F"

        status = "Critical"

    return {
        "score": score,
        "grade": grade,
        "status": status,
    }


# ==============================================================================
# Dataset Summary
# ==============================================================================


def generate_summary(df):

    feature_types = detect_feature_types(df)

    memory = estimate_memory_saving(df)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "memory": memory["current"],
        "optimized_memory": memory["optimized"],
        "memory_saved": memory["saved"],
        "memory_saved_percent": memory["saving_percent"],
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percent": percentage(
            int(df.isna().sum().sum()), df.shape[0] * df.shape[1]
        ),
        "numeric_columns": len(feature_types["numeric"]),
        "categorical_columns": len(feature_types["categorical"]),
        "datetime_columns": len(feature_types["datetime"]),
        "boolean_columns": len(feature_types["boolean"]),
        "text_columns": len(feature_types["text"]),
    }


# ==============================================================================
# AI Recommendations (Basic)
# ==============================================================================


def generate_basic_recommendations(df):

    recommendations = []

    if df.duplicated().sum() > 0:

        recommendations.append("Duplicate rows detected. Removing them is recommended.")

    if df.isna().sum().sum() > 0:

        recommendations.append(
            "Missing values detected. Apply suitable imputation before training."
        )

    ids = detect_identifier_columns(df)

    if ids:

        recommendations.append(f"{len(ids)} possible identifier column(s) detected.")

    constants = detect_constant_columns(df)

    if constants:

        recommendations.append(f"{len(constants)} constant column(s) detected.")

    datetime_columns = detect_datetime_columns(df)

    if datetime_columns:

        recommendations.append(f"{len(datetime_columns)} datetime column(s) detected.")

    if len(recommendations) == 0:

        recommendations.append("Dataset appears clean and ready for preprocessing.")

    return recommendations


# ==============================================================================
# Missing Value Analysis
# ==============================================================================


def analyze_missing_values(df):

    missing_summary = []

    total_rows = len(df)

    for column in df.columns:

        missing = int(df[column].isna().sum())

        if missing == 0:
            continue

        percent = percentage(
            missing,
            total_rows,
        )

        if is_numeric_dtype(df[column]):

            recommendation = "Fill using Median"

        elif is_bool_dtype(df[column]):

            recommendation = "Fill using Mode"

        else:

            recommendation = "Fill using Mode"

        severity = "Low"

        if percent >= 50:

            severity = "High"

        elif percent >= 20:

            severity = "Medium"

        missing_summary.append(
            {
                "column": column,
                "missing": missing,
                "percent": percent,
                "severity": severity,
                "recommendation": recommendation,
            }
        )

    missing_summary.sort(
        key=lambda x: x["percent"],
        reverse=True,
    )

    return missing_summary


# ==============================================================================
# Duplicate Analysis
# ==============================================================================


def analyze_duplicates(df):

    duplicate_rows = int(df.duplicated().sum())

    duplicate_percent = percentage(
        duplicate_rows,
        len(df),
    )

    return {
        "rows": duplicate_rows,
        "percent": duplicate_percent,
        "recommended": duplicate_rows > 0,
    }


# ==============================================================================
# Numeric Column Statistics
# ==============================================================================


def numeric_statistics(df):

    statistics = {}

    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in numeric_columns:

        series = df[column]

        statistics[column] = {
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "mean": round(series.mean(), 4) if series.count() else None,
            "median": round(series.median(), 4) if series.count() else None,
            "std": round(series.std(), 4) if series.count() else None,
            "variance": round(series.var(), 4) if series.count() else None,
            "minimum": round(series.min(), 4) if series.count() else None,
            "maximum": round(series.max(), 4) if series.count() else None,
            "q1": round(series.quantile(0.25), 4) if series.count() else None,
            "q3": round(series.quantile(0.75), 4) if series.count() else None,
        }

    return statistics


# ==============================================================================
# Categorical Statistics
# ==============================================================================


def categorical_statistics(df):

    statistics = {}

    object_columns = df.select_dtypes(include=["object", "category"]).columns

    for column in object_columns:

        series = df[column]

        mode = None

        if not series.mode().empty:

            mode = str(series.mode().iloc[0])

        statistics[column] = {
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
            "top": mode,
            "top_frequency": (
                int(series.value_counts(dropna=False).iloc[0]) if len(series) > 0 else 0
            ),
        }

    return statistics


# ==============================================================================
# IQR Outlier Detection
# ==============================================================================


def detect_outliers(df):

    outliers = {}

    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 5:

            continue

        q1 = series.quantile(0.25)

        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        count = int(((series < lower) | (series > upper)).sum())

        outliers[column] = {
            "count": count,
            "percent": percentage(
                count,
                len(series),
            ),
            "lower_limit": round(lower, 4),
            "upper_limit": round(upper, 4),
        }

    return outliers


# ==============================================================================
# Column Profiling
# ==============================================================================


def generate_column_profiles(df):

    profiles = []

    outlier_info = detect_outliers(df)

    for column in df.columns:

        series = df[column]

        profile = {
            "name": column,
            "dtype": str(series.dtype),
            "rows": len(series),
            "missing": int(series.isna().sum()),
            "missing_percent": percentage(
                int(series.isna().sum()),
                len(series),
            ),
            "unique": int(series.nunique()),
            "memory_mb": round(
                series.memory_usage(deep=True) / 1024 / 1024,
                3,
            ),
            "constant": series.nunique(dropna=False) <= 1,
            "identifier": column in detect_identifier_columns(df),
        }

        if is_numeric_dtype(series):

            profile.update(
                {
                    "minimum": round(series.min(), 4) if series.count() else None,
                    "maximum": round(series.max(), 4) if series.count() else None,
                    "mean": round(series.mean(), 4) if series.count() else None,
                    "median": round(series.median(), 4) if series.count() else None,
                    "std": round(series.std(), 4) if series.count() else None,
                    "outliers": outlier_info.get(
                        column,
                        {},
                    ).get(
                        "count",
                        0,
                    ),
                }
            )

        else:

            mode = None

            if not series.mode().empty:

                mode = str(series.mode().iloc[0])

            profile.update(
                {
                    "top_value": mode,
                    "top_frequency": (
                        int(series.value_counts(dropna=False).iloc[0])
                        if len(series)
                        else 0
                    ),
                }
            )

        profiles.append(profile)

    return profiles


# ==============================================================================
# Data Quality Warnings
# ==============================================================================


def generate_quality_warnings(df):

    warnings = []

    missing_analysis = analyze_missing_values(df)
    duplicate_analysis = analyze_duplicates(df)
    outliers = detect_outliers(df)

    if duplicate_analysis["rows"] > 0:

        warnings.append(
            {
                "type": "warning",
                "title": "Duplicate Rows",
                "description": f"{duplicate_analysis['rows']} duplicate rows detected.",
                "action": "Remove duplicate rows.",
            }
        )

    for item in missing_analysis:

        if item["severity"] == "High":

            warnings.append(
                {
                    "type": "danger",
                    "title": item["column"],
                    "description": f"{item['percent']}% values are missing.",
                    "action": item["recommendation"],
                }
            )

    for column in detect_constant_columns(df):

        warnings.append(
            {
                "type": "warning",
                "title": column,
                "description": "Constant column detected.",
                "action": "Remove column.",
            }
        )

    for column in detect_identifier_columns(df):

        warnings.append(
            {
                "type": "info",
                "title": column,
                "description": "Possible Identifier Column.",
                "action": "Exclude from Machine Learning.",
            }
        )

    for column, info in outliers.items():

        if info["percent"] >= 5:

            warnings.append(
                {
                    "type": "warning",
                    "title": column,
                    "description": f"{info['count']} outliers detected.",
                    "action": "Consider clipping or removing.",
                }
            )

    return warnings


# ==============================================================================
# Cleaning Recommendations
# ==============================================================================


def generate_cleaning_plan(df):

    plan = []

    if df.duplicated().sum() > 0:

        plan.append(
            {
                "step": 1,
                "task": "Remove Duplicate Rows",
                "enabled": True,
                "count": int(df.duplicated().sum()),
            }
        )

    missing = analyze_missing_values(df)

    if missing:

        plan.append(
            {
                "step": 2,
                "task": "Handle Missing Values",
                "enabled": True,
                "count": len(missing),
            }
        )

    constants = detect_constant_columns(df)

    if constants:

        plan.append(
            {
                "step": 3,
                "task": "Remove Constant Columns",
                "enabled": True,
                "count": len(constants),
            }
        )

    identifiers = detect_identifier_columns(df)

    if identifiers:

        plan.append(
            {
                "step": 4,
                "task": "Remove Identifier Columns",
                "enabled": False,
                "count": len(identifiers),
            }
        )

    datetime_columns = detect_datetime_columns(df)

    if datetime_columns:

        plan.append(
            {
                "step": 5,
                "task": "Convert Datetime Columns",
                "enabled": True,
                "count": len(datetime_columns),
            }
        )

    return plan


# ==============================================================================
# Correlation Matrix
# ==============================================================================


def generate_correlation(df):

    numeric = df.select_dtypes(include=np.number)

    if numeric.shape[1] < 2:

        return []

    corr = numeric.corr().round(3)

    correlation = []

    for i in corr.columns:

        for j in corr.columns:

            if i >= j:
                continue

            value = corr.loc[i, j]

            if abs(value) >= 0.70:

                correlation.append(
                    {
                        "column_1": i,
                        "column_2": j,
                        "correlation": float(value),
                    }
                )

    correlation.sort(
        key=lambda x: abs(x["correlation"]),
        reverse=True,
    )

    return correlation


# ==============================================================================
# Dataset Readiness
# ==============================================================================


def dataset_readiness(df):

    score = calculate_health_score(df)["score"]

    if score >= 90:

        status = "Ready"

        color = "green"

    elif score >= 75:

        status = "Needs Minor Cleaning"

        color = "yellow"

    elif score >= 60:

        status = "Needs Cleaning"

        color = "orange"

    else:

        status = "Poor Quality"

        color = "red"

    return {
        "score": score,
        "status": status,
        "color": color,
    }


# ==============================================================================
# Feature Engineering Suggestions
# ==============================================================================


def feature_engineering_suggestions(df):

    suggestions = []

    feature_types = detect_feature_types(df)

    if feature_types["categorical"]:

        suggestions.append("Encode categorical columns.")

    if feature_types["datetime"]:

        suggestions.append("Extract Year, Month, Day from datetime.")

    if feature_types["numeric"]:

        suggestions.append("Scale numeric features before training.")

    if detect_identifier_columns(df):

        suggestions.append("Remove identifier columns.")

    if detect_outliers(df):

        suggestions.append("Handle outliers before model training.")

    return suggestions


# ==============================================================================
# Processing Log
# ==============================================================================


def generate_processing_log(df):

    log = []

    log.append("Dataset Loaded Successfully")

    log.append(f"Rows Detected : {len(df)}")

    log.append(f"Columns Detected : {len(df.columns)}")

    log.append(f"Missing Cells : {df.isna().sum().sum()}")

    log.append(f"Duplicate Rows : {df.duplicated().sum()}")

    log.append(f"Identifier Columns : {len(detect_identifier_columns(df))}")

    log.append(f"Constant Columns : {len(detect_constant_columns(df))}")

    log.append(f"Datetime Columns : {len(detect_datetime_columns(df))}")

    log.append(f"Outlier Columns : {len(detect_outliers(df))}")

    return log


# ==============================================================================
# Cleaning Options (Default UI Values)
# ==============================================================================


def default_cleaning_options():

    return {
        "remove_duplicates": True,
        "remove_constant_columns": True,
        "remove_identifier_columns": False,
        "convert_datetime": True,
        "handle_missing_numeric": "median",
        "handle_missing_categorical": "mode",
        "outlier_method": "iqr",
        "encoding": "onehot",
        "scaling": "none",
        "normalize": False,
        "feature_engineering": True,
    }


# ==============================================================================
# Apply Cleaning Pipeline
# ==============================================================================


def apply_cleaning_pipeline(df, options=None):

    if options is None:
        options = default_cleaning_options()

    cleaned_df = df.copy()

    log = []

    # --------------------------------------------------------------------------
    # Remove Duplicate Rows
    # --------------------------------------------------------------------------

    if options.get("remove_duplicates", True):

        before = len(cleaned_df)

        cleaned_df = cleaned_df.drop_duplicates()

        removed = before - len(cleaned_df)

        log.append(f"Removed {removed} duplicate rows.")

    # --------------------------------------------------------------------------
    # Remove Constant Columns
    # --------------------------------------------------------------------------

    if options.get("remove_constant_columns", True):

        constant_columns = detect_constant_columns(cleaned_df)

        if constant_columns:

            cleaned_df = cleaned_df.drop(
                columns=constant_columns,
                errors="ignore",
            )

        log.append(f"Removed {len(constant_columns)} constant column(s).")

    # --------------------------------------------------------------------------
    # Remove Identifier Columns
    # --------------------------------------------------------------------------

    if options.get("remove_identifier_columns", False):

        identifier_columns = detect_identifier_columns(cleaned_df)

        if identifier_columns:

            cleaned_df = cleaned_df.drop(
                columns=identifier_columns,
                errors="ignore",
            )

        log.append(f"Removed {len(identifier_columns)} identifier column(s).")

    # --------------------------------------------------------------------------
    # Convert Datetime Columns
    # --------------------------------------------------------------------------

    if options.get("convert_datetime", True):

        datetime_columns = detect_datetime_columns(cleaned_df)

        for column in datetime_columns:

            cleaned_df[column] = pd.to_datetime(
                cleaned_df[column],
                errors="coerce",
                infer_datetime_format=True,
            )

        log.append(f"Converted {len(datetime_columns)} datetime column(s).")

    # --------------------------------------------------------------------------
    # Handle Missing Numeric Values
    # --------------------------------------------------------------------------

    numeric_method = options.get(
        "handle_missing_numeric",
        "median",
    ).lower()

    numeric_columns = cleaned_df.select_dtypes(include=np.number).columns

    for column in numeric_columns:

        if cleaned_df[column].isna().sum() == 0:
            continue

        if numeric_method == "mean":

            value = cleaned_df[column].mean()

        elif numeric_method == "median":

            value = cleaned_df[column].median()

        elif numeric_method == "mode":

            mode = cleaned_df[column].mode()

            value = mode.iloc[0] if not mode.empty else 0

        else:

            value = cleaned_df[column].median()

        cleaned_df[column] = cleaned_df[column].fillna(value)

    log.append(f"Numeric missing values handled using {numeric_method}.")

    # --------------------------------------------------------------------------
    # Handle Missing Categorical Values
    # --------------------------------------------------------------------------

    categorical_method = options.get(
        "handle_missing_categorical",
        "mode",
    ).lower()

    categorical_columns = cleaned_df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        if cleaned_df[column].isna().sum() == 0:
            continue

        if categorical_method == "constant":

            value = "Unknown"

        elif categorical_method == "drop":

            cleaned_df = cleaned_df.dropna(subset=[column])

            continue

        else:

            mode = cleaned_df[column].mode()

            value = mode.iloc[0] if not mode.empty else "Unknown"

        cleaned_df[column] = cleaned_df[column].fillna(value)

    log.append(f"Categorical missing values handled using {categorical_method}.")

    # --------------------------------------------------------------------------
    # Export Cleaned Dataset
    # --------------------------------------------------------------------------

    exported_file = None

    export_excel = options.get(
        "export_excel",
        False,
    )

    if export_excel:

        export_path = options.get(
            "export_path",
            os.getcwd(),
        )

        export_filename = options.get(
            "export_filename",
            "Cleaned_Dataset.xlsx",
        )

        os.makedirs(
            export_path,
            exist_ok=True,
        )

        exported_file = os.path.join(
            export_path,
            export_filename,
        )

        cleaned_df.to_excel(
            exported_file,
            index=False,
        )

        log.append(f"Cleaned dataset exported to '{exported_file}'.")

    # --------------------------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------------------------

    result = {
        "dataframe": cleaned_df,
        "summary": generate_summary(cleaned_df),
        "health": calculate_health_score(cleaned_df),
        "readiness": dataset_readiness(cleaned_df),
        "recommendations": generate_basic_recommendations(cleaned_df),
        "processing_log": log,
        "exported_file": exported_file,
    }

    return result
