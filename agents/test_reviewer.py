from reviewer import execute_and_review

def test_execute_and_review(dummy_df):
    # This code drops column D
    python_code = "df = df.drop(columns=['D'])"
    
    review, cleaned_df = execute_and_review(dummy_df, python_code)
    
    assert review["execution_success"] is True
    assert "D" not in cleaned_df.columns
    assert cleaned_df.shape[1] == 3 # A, B, C

def test_execute_and_review_failure(dummy_df):
    # This code will crash
    python_code = "df = df.drop(columns=['NON_EXISTENT_COLUMN'])"
    
    review, cleaned_df = execute_and_review(dummy_df, python_code)
    
    assert review["execution_success"] is False
    assert "error_message" in review