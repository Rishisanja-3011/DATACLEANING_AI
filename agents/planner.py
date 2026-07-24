from typing import Dict, Any


def create_plan(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a deterministic preprocessing plan from the Inspector report.
    """

    plan = {
        "dataset_actions": [],
        "column_actions": {}
    }

    total_rows = report["shape"]["rows"]
    data_types = report["data_types"]

    # ==========================
    # Dataset-level actions
    # ==========================

    if report["duplicate_rows"] > 0:
        plan["dataset_actions"].append({
            "action": "remove_duplicates",
            "reason": f"{report['duplicate_rows']} duplicate rows detected."
        })

    # ==========================
    # Column-level actions
    # ==========================

    for col in report["column_names"]:

        actions = []

        dtype = data_types[col]

        missing = report["missing_values"][col]["count"]

        missing_percentage = report["missing_values"][col]["percentage"]

        unique_values = report["unique_values"][col]

        # --------------------------
        # Drop constant columns
        # --------------------------

        if col in report["constant_columns"]:

            actions.append({
                "action": "drop_column",
                "reason": "Constant column."
            })

            plan["column_actions"][col] = actions
            continue

        # --------------------------
        # Drop empty columns
        # --------------------------

        if missing == total_rows:

            actions.append({
                "action": "drop_column",
                "reason": "Column is completely empty."
            })

            plan["column_actions"][col] = actions
            continue

        # --------------------------
        # Drop ID columns
        # --------------------------

        col_lower = col.lower()

        if (
            col_lower == "id"
            or col_lower.endswith("_id")
            or col_lower.endswith("id")
        ):

            actions.append({
                "action": "drop_column",
                "reason": "Identifier column."
            })

            plan["column_actions"][col] = actions
            continue

        # --------------------------
        # Datetime Features
        # --------------------------

        if col in report["possible_datetime_columns"]:

            actions.append({
                "action": "extract_datetime_features",
                "reason": "Detected datetime column."
            })

        # --------------------------
        # Missing Values
        # --------------------------

        if missing > 0:

            if col in report["numeric_columns"]:

                actions.append({
                    "action": "median_imputation",
                    "reason": f"{missing_percentage}% missing values in numeric column."
                })

            elif col in report["categorical_columns"]:

                actions.append({
                    "action": "most_frequent_imputation",
                    "reason": f"{missing_percentage}% missing values in categorical column."
                })

        # --------------------------
        # Encoding
        # --------------------------

        # Do not encode columns that are being treated as datetime.
        # The original column will be replaced by extracted
        # year/month/day features.
        if (
            col in report["categorical_columns"]
            and col not in report["possible_datetime_columns"]
        ):

            if col in report["high_cardinality_columns"]:

                actions.append({
                    "action": "label_encoding",
                    "reason": "High-cardinality categorical column."
                })

            else:

                actions.append({
                    "action": "onehot_encoding",
                    "reason": "Low-cardinality categorical column."
                })

        # --------------------------
        # Outlier Handling
        # --------------------------

        if col in report["numeric_columns"]:

            outlier_count = report["outliers"][col]

            if outlier_count > 0:

                actions.append({
                    "action": "clip_outliers",
                    "reason": f"{outlier_count} outliers detected."
                })

        # --------------------------
        # Scaling
        # --------------------------

        if col in report["numeric_columns"]:

            actions.append({
                "action": "standard_scaling",
                "reason": "Numeric feature."
            })

        if actions:
            plan["column_actions"][col] = actions

    return plan