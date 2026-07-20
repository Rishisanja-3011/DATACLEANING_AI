import traceback
import pandas as pd


def execute_and_review(df, python_code):
    """
    Execute generated preprocessing code and review the cleaned dataframe.
    """

    local_env = {
        "df": df.copy(),
        "pd": pd
    }

    original_rows = len(df)
    original_columns = list(df.columns)
    original_missing = int(df.isnull().sum().sum())
    original_duplicates = int(df.duplicated().sum())

    try:

        # -------------------------
        # Execute Generated Code
        # -------------------------

        exec(python_code, globals(), local_env)

        cleaned_df = local_env["df"]

        if not isinstance(cleaned_df, pd.DataFrame):
            raise TypeError("Generated code did not return a pandas DataFrame.")

        # -------------------------
        # Metrics
        # -------------------------

        final_missing = int(cleaned_df.isnull().sum().sum())
        final_duplicates = int(cleaned_df.duplicated().sum())

        # -------------------------
        # Validation Checks
        # -------------------------

        validations = {
            "is_dataframe": isinstance(cleaned_df, pd.DataFrame),
            "rows_remaining": len(cleaned_df),
            "columns_remaining": len(cleaned_df.columns),
            "missing_values_reduced": final_missing <= original_missing,
            "duplicates_reduced": final_duplicates <= original_duplicates,
            "column_count_valid": len(cleaned_df.columns) > 0,
            "row_count_valid": len(cleaned_df) > 0,
        }

        # -------------------------
        # Quality Score
        # -------------------------

        score = 0

        for passed in validations.values():

            if isinstance(passed, bool) and passed:
                score += 1

        review = {
            "execution_success": True,
            "quality_score": f"{score}/{len([v for v in validations.values() if isinstance(v,bool)])}",
            "validations": validations,
            "original_shape": df.shape,
            "final_shape": cleaned_df.shape,
            "original_missing": original_missing,
            "remaining_missing": final_missing,
            "original_duplicates": original_duplicates,
            "remaining_duplicates": final_duplicates,
            "columns": list(cleaned_df.columns),
        }

        return review, cleaned_df

    except Exception as e:

        review = {
            "execution_success": False,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "quality_score": "0/7"
        }

        return review, df