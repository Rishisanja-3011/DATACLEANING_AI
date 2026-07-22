import traceback
from typing import Any, Dict, Tuple

import pandas as pd


def execute_and_review(
    df: pd.DataFrame,
    python_code: str
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Execute generated preprocessing code and validate the resulting DataFrame.

    The original DataFrame is never modified directly.
    """

    original_df = df.copy()

    # =========================
    # ORIGINAL METRICS
    # =========================

    original_rows = len(original_df)
    original_columns = len(original_df.columns)
    original_missing = int(original_df.isnull().sum().sum())
    original_duplicates = int(original_df.duplicated().sum())

    # =========================
    # EXECUTION ENVIRONMENT
    # =========================

    local_env = {
        "df": original_df.copy(),
        "pd": pd,
    }

    try:

        # =========================
        # EXECUTE GENERATED CODE
        # =========================

        exec(
            python_code,
            {"__builtins__": __builtins__},
            local_env
        )

        cleaned_df = local_env.get("df")

        if not isinstance(cleaned_df, pd.DataFrame):
            raise TypeError(
                "Generated preprocessing code did not leave a "
                "pandas DataFrame in variable 'df'."
            )

        # =========================
        # FINAL METRICS
        # =========================

        final_rows = len(cleaned_df)
        final_columns = len(cleaned_df.columns)

        final_missing = int(
            cleaned_df.isnull().sum().sum()
        )

        final_duplicates = int(
            cleaned_df.duplicated().sum()
        )

        # =========================
        # METRICS
        # =========================

        metrics = {
            "original_rows": original_rows,
            "final_rows": final_rows,

            "original_columns": original_columns,
            "final_columns": final_columns,

            "original_missing": original_missing,
            "remaining_missing": final_missing,

            "original_duplicates": original_duplicates,
            "remaining_duplicates": final_duplicates,
        }

        # =========================
        # VALIDATIONS
        # =========================

        validations = {
            "is_dataframe": True,

            "row_count_valid":
                final_rows > 0,

            "column_count_valid":
                final_columns > 0,

            "missing_values_not_increased":
                final_missing <= original_missing,

            "duplicates_not_increased":
                final_duplicates <= original_duplicates,
        }

        # =========================
        # QUALITY SCORE
        # =========================

        passed_checks = sum(
            1
            for passed in validations.values()
            if passed
        )

        total_checks = len(validations)

        validation_success = all(
            validations.values()
        )

        # =========================
        # REVIEW REPORT
        # =========================

        review = {
            "execution_success": True,

            "validation_success":
                validation_success,

            "pipeline_success":
                validation_success,

            "quality_score":
                f"{passed_checks}/{total_checks}",

            "metrics":
                metrics,

            "validations":
                validations,

            "original_shape":
                original_df.shape,

            "final_shape":
                cleaned_df.shape,

            "columns":
                list(cleaned_df.columns),

            "error_message":
                None,
        }

        return review, cleaned_df

    except Exception as exc:

        # =========================
        # EXECUTION FAILURE
        # =========================

        review = {
            "execution_success": False,

            "validation_success": False,

            "pipeline_success": False,

            "quality_score": "0/5",

            "metrics": {
                "original_rows":
                    original_rows,

                "original_columns":
                    original_columns,

                "original_missing":
                    original_missing,

                "original_duplicates":
                    original_duplicates,
            },

            "validations": {},

            "original_shape":
                original_df.shape,

            "final_shape":
                None,

            "columns":
                list(original_df.columns),

            "error_message":
                str(exc),

            "traceback":
                traceback.format_exc(),
        }

        # Return original data on failure
        return review, original_df