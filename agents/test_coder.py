import ast

from agents.coder import generate_preprocessing_code
from agents.inspector import inspect_data
from agents.planner import create_plan


def test_generate_preprocessing_code(dummy_df):
    """
    Integration test for the Coder Agent.
    """

    report = inspect_data(dummy_df)
    plan = create_plan(report)

    python_code = generate_preprocessing_code(
        dummy_df,
        plan,
        report
    )

    assert isinstance(python_code, str)
    assert python_code.strip()
    assert "df" in python_code

    # Generated output must be valid Python
    ast.parse(python_code)