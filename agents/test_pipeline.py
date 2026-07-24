from agents.graph_pipeline import app


def test_graph_pipeline(dummy_df):
    """
    End-to-end integration test for the complete agent pipeline.

    Flow:
    Inspector -> Planner -> Coder -> Reviewer
    """

    # Run complete LangGraph pipeline
    result = app.invoke({
        "df": dummy_df
    })

    # -------------------------
    # State validation
    # -------------------------

    assert "report" in result
    assert "plan" in result
    assert "python_code" in result
    assert "review" in result
    assert "cleaned_df" in result

    # -------------------------
    # Inspector validation
    # -------------------------

    assert isinstance(result["report"], dict)
    assert "shape" in result["report"]

    # -------------------------
    # Planner validation
    # -------------------------

    assert isinstance(result["plan"], dict)
    assert "dataset_actions" in result["plan"]
    assert "column_actions" in result["plan"]

    # -------------------------
    # Coder validation
    # -------------------------

    assert isinstance(result["python_code"], str)
    assert result["python_code"].strip()
    assert "df" in result["python_code"]

    # -------------------------
    # Reviewer validation
    # -------------------------

    review = result["review"]

    assert review["execution_success"] is True
    assert review["validation_success"] is True
    assert review["pipeline_success"] is True

    assert "metrics" in review
    assert "validations" in review

    # -------------------------
    # Cleaned DataFrame
    # -------------------------

    cleaned_df = result["cleaned_df"]

    assert cleaned_df is not None
    assert cleaned_df.shape[0] > 0
    assert cleaned_df.shape[1] > 0

    # The dummy dataset contains duplicates,
    # so duplicate count should not increase.
    assert (
        cleaned_df.duplicated().sum()
        <= dummy_df.duplicated().sum()
    )