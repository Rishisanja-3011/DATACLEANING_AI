from graph_pipeline import app

def test_graph_pipeline(dummy_df):
    result = app.invoke({"df": dummy_df})
    
    assert "review" in result
    assert result["review"]["execution_success"] is True
    
    cleaned_df = result["cleaned_df"]
    
    # Original shape was (5,4). With duplicates removed and D dropped, it should be smaller
    assert cleaned_df.shape[0] < 5
    assert cleaned_df.shape[1] <= 4 # Might have dummy columns or dropped cols