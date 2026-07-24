import pandas as pd


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Perform deterministic inspection of a dataset.

    Returns a structured report that downstream agents
    (Planner, Reviewer) can consume.
    """

    report = {}

    # ==========================
    # Basic Information
    # ==========================

    report["shape"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }

    report["column_names"] = df.columns.tolist()

    report["memory_usage_mb"] = round(
        df.memory_usage(deep=True).sum() / (1024 * 1024), 2
    )

    # ==========================
    # Data Types
    # ==========================

    report["data_types"] = df.dtypes.astype(str).to_dict()

    report["numeric_columns"] = (
        df.select_dtypes(include="number").columns.tolist()
    )

    report["categorical_columns"] = (
        df.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
    )   

    report["datetime_columns"] = (
        df.select_dtypes(include=["datetime"]).columns.tolist()
    )

    report["boolean_columns"] = (
        df.select_dtypes(include=["bool"]).columns.tolist()
    )

    # ==========================
    # Missing Values
    # ==========================

    missing = {}

    for col in df.columns:
        count = int(df[col].isnull().sum())

        missing[col] = {
            "count": count,
            "percentage": round((count / len(df)) * 100, 2),
        }

    report["missing_values"] = missing

    # ==========================
    # Duplicate Rows
    # ==========================

    report["duplicate_rows"] = int(df.duplicated().sum())

    # ==========================
    # Duplicate Columns
    # ==========================

    duplicate_columns = []

    cols = df.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                duplicate_columns.append(cols[j])

    report["duplicate_columns"] = duplicate_columns

    # ==========================
    # Constant Columns
    # ==========================

    constant_columns = [
        col for col in df.columns if df[col].nunique(dropna=False) <= 1
    ]

    report["constant_columns"] = constant_columns

    # ==========================
    # Unique Values
    # ==========================

    report["unique_values"] = df.nunique(dropna=False).to_dict()

    # ==========================
    # High Cardinality
    # ==========================

    high_cardinality = []

    for col in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns:      

        ratio = df[col].nunique() / max(len(df), 1)

        if ratio > 0.90:
            high_cardinality.append(col)

    report["high_cardinality_columns"] = high_cardinality

    # ==========================
    # Numeric Summary
    # ==========================

    numeric_summary = {}

    for col in report["numeric_columns"]:

        numeric_summary[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }

    report["numeric_summary"] = numeric_summary

    # ==========================
    # Outlier Detection (IQR)
    # ==========================

    outliers = {}

    for col in report["numeric_columns"]:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(((df[col] < lower) | (df[col] > upper)).sum())

        outliers[col] = count

    report["outliers"] = outliers

    # ==========================
    # Empty Strings
    # ==========================

    empty_strings = {}

    for col in report["categorical_columns"]:

        empty_strings[col] = int(
            df[col]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

    report["empty_strings"] = empty_strings

    # ==========================
    # Date Detection
    # ==========================

    possible_dates = []

    for col in report["categorical_columns"]:

        sample = df[col].dropna()

        if sample.empty:
            continue

        # Convert values to strings
        sample = sample.astype(str)

        # Basic heuristic:
        # only attempt datetime parsing when values
        # contain common date separators.
        looks_like_date = sample.str.contains(
            r"[-/]",
            regex=True
        ).mean()

        if looks_like_date < 0.8:
            continue

        try:

            converted = pd.to_datetime(
                sample,
                errors="coerce"
            )

            success_ratio = converted.notna().mean()

            if success_ratio >= 0.8:
                possible_dates.append(col)

        except (ValueError, TypeError):
            continue

    report["possible_datetime_columns"] = possible_dates

    return report