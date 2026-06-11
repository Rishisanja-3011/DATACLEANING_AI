from coder import generate_preprocessing_code
from inspector import inspect_data
from planner import create_plan

def test_generate_preprocessing_code(dummy_df):
    report = inspect_data(dummy_df)
    plan = create_plan(report)
    
    python_code = generate_preprocessing_code(dummy_df, plan, report)
    
    assert isinstance(python_code, str)
    assert len(python_code) > 0
    assert "df" in python_code