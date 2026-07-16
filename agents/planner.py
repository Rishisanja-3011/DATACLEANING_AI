def create_plan(report):

    plan = {}
    total_rows = report["shape"][0]

    # Phase 1: Identifiers, Dates, and Missing Values
    for col, missing in report["missing_values"].items():
        dtype = report["columns"][col]
        col_lower = col.lower()

        # 1. Drop fully empty columns
        if missing == total_rows:
            plan[col] = "drop_column"
            continue

        # 2. Identifiers hold no predictive value
        if col_lower.endswith("_id") or col_lower == "id":
            plan[col] = "drop_column"
            continue

        # 3. Dates need feature extraction
        if "date" in col_lower or "timestamp" in col_lower or dtype == "datetime64[ns]":
            plan[col] = "extract_datetime_features"
            continue

        # 4. Impute Missing Values
        if missing > 0:
            if dtype in ["int64", "float64"]:
                plan[col] = "mean_imputation"
            elif dtype in ["object", "str", "string"]:
                plan[col] = "most_frequent_imputation"

    # Phase 2: Duplicates
    if report["duplicates"] > 0:
        plan["duplicates"] = "remove_duplicates"

    # Phase 3: Encoding Categoricals
    for col, dtype in report["columns"].items():
        if col in plan and plan[col] in ["drop_column", "extract_datetime_features"]:
            continue

        unique_count = report.get("unique_values", {}).get(col, 0)

        if dtype in ["object", "str", "string"] and col not in plan:
            # Low-cardinality columns
            if unique_count <= 10:
                plan[col] = "onehot_encoding"
            # High-cardinality columns
            else:
                plan[col] = "label_encoding"

    # Phase 4: Scaling Numerics
    for col, dtype in report["columns"].items():
        if col in plan and plan[col] in ["drop_column", "extract_datetime_features"]:
            continue

        if dtype in ["int64", "float64"] and col not in plan:
            plan[col] = "standard_scaling"

    return plan