from inspector import inspect_data
from planner import create_plan

def test_create_plan(dummy_df):
    report = inspect_data(dummy_df)
    plan = create_plan(report)
    
    assert isinstance(plan, dict)
    
    # Check if completely missing column D is dropped
    assert "D" in plan
    assert plan["D"] == "drop_column"
    
    # Check if duplicates are removed
    assert "duplicates" in plan
    assert plan["duplicates"] == "remove_duplicates"