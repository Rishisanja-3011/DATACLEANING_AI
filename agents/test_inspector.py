from agents.inspector import inspect_data


def test_inspect_data(dummy_df):
    report = inspect_data(dummy_df)

    assert isinstance(report, dict)

    # Dataset shape
    assert "shape" in report
    assert report["shape"]["rows"] == 5
    assert report["shape"]["columns"] == 6

    # Columns
    assert report["column_names"] == [
        "user_id",
        "start_date",
        "A",
        "B",
        "C",
        "D"
    ]

    # Data type groups
    assert "numeric_columns" in report
    assert "categorical_columns" in report

    # Data quality information
    assert "missing_values" in report
    assert "duplicate_rows" in report
    assert "constant_columns" in report
    assert "outliers" in report

    # Dummy dataset contains one duplicate row
    assert report["duplicate_rows"] == 1

    # D is completely empty and therefore constant
    assert "D" in report["constant_columns"]