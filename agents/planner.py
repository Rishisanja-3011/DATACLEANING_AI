def create_plan(report):

    plan = {}

    total_rows = report["shape"][0]

    for col, missing in report["missing_values"].items():

        dtype = report["columns"][col]

        # Drop fully empty columns
        if missing == total_rows:
            plan[col] = "drop_column"

        # Missing numeric values
        elif missing > 0 and dtype in ["int64", "float64"]:
            plan[col] = "mean_imputation"

        # Missing categorical values
        elif missing > 0 and dtype == "object":
            plan[col] = "most_frequent_imputation"

    # Remove duplicates
    if report["duplicates"] > 0:
        plan["duplicates"] = "remove_duplicates"

    # One-hot encoding for categorical columns
    for col, dtype in report["columns"].items():

        unique_count = report.get("unique_values", {}).get(col, 0)

        if dtype == "object" and col not in plan:

            # Low-cardinality columns
            if unique_count <= 10:
                plan[col] = "onehot_encoding"

            # High-cardinality columns
            else:
                plan[col] = "label_encoding"

    # Scaling numeric columns
    for col, dtype in report["columns"].items():

        if dtype in ["int64", "float64"] and col not in plan:
            plan[col] = "standard_scaling"

    return plan