from agents.reviewer import execute_and_review


def test_execute_and_review(dummy_df):
    """
    Test successful preprocessing execution.
    """

    # Drop column D
    python_code = "df = df.drop(columns=['D'])"

    review, cleaned_df = execute_and_review(
        dummy_df,
        python_code
    )

    # Execution should succeed
    assert review["execution_success"] is True

    # Validation should succeed
    assert review["validation_success"] is True

    # Entire pipeline step should succeed
    assert review["pipeline_success"] is True

    # Column D should be removed
    assert "D" not in cleaned_df.columns

    # Expected remaining columns:
    # user_id, start_date, A, B, C
    assert cleaned_df.shape[1] == 5

    # Metrics should be available
    assert "metrics" in review
    assert "validations" in review

    assert review["metrics"]["original_columns"] == 6
    assert review["metrics"]["final_columns"] == 5

    # No execution error
    assert review["error_message"] is None


def test_execute_and_review_failure(dummy_df):
    """
    Test generated code that raises an exception.
    """

    python_code = (
        "df = df.drop(columns=['NON_EXISTENT_COLUMN'])"
    )

    review, cleaned_df = execute_and_review(
        dummy_df,
        python_code
    )

    # Execution should fail
    assert review["execution_success"] is False

    assert review["validation_success"] is False

    assert review["pipeline_success"] is False

    # Error information should exist
    assert review["error_message"]

    assert "traceback" in review

    # Reviewer should return the original DataFrame
    assert cleaned_df.equals(dummy_df)