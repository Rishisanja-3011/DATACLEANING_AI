from inspector import inspect_data

def test_inspect_data(dummy_df):
    report = inspect_data(dummy_df)
    
    assert "shape" in report
    assert report["shape"] == (5, 4)
    
    assert "missing_values" in report
    assert report["missing_values"]["A"] == 1
    assert report["missing_values"]["D"] == 5
    
    assert "duplicates" in report
    assert report["duplicates"] == 1
