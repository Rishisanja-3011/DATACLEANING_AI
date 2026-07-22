import ast

from coder import generate_preprocessing_code
from inspector import inspect_data
from planner import create_plan


def test_generate_preprocessing_code(dummy_df):
    """
    Integration test for the Coder Agent.

    Verifies that the Coder:
    1. Accepts the Inspector report.
    2. Accepts the Planner plan.
    3. Returns non-empty Python code.
    4. Generates syntactically valid Python.
    5. References the working DataFrame `df`.
    """

    # Arrange
    report = inspect_data(dummy_df)
    plan = create_plan(report)

    # Act
    python_code = generate_preprocessing_code(
        dummy_df,
        plan,
        report
    )

    # Assert
    assert isinstance(python_code, str)

    assert python_code.strip()

    assert "df" in python_code

    # Generated output must be valid Python
    ast.parse(python_code)