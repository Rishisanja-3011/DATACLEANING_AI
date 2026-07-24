from agents.inspector import inspect_data
from agents.planner import create_plan


def test_create_plan(dummy_df):
    report = inspect_data(dummy_df)

    plan = create_plan(report)

    assert isinstance(plan, dict)

    # -------------------------
    # Plan structure
    # -------------------------

    assert "dataset_actions" in plan
    assert "column_actions" in plan

    assert isinstance(plan["dataset_actions"], list)
    assert isinstance(plan["column_actions"], dict)

    # -------------------------
    # Duplicate handling
    # -------------------------

    dataset_action_names = [
        action["action"]
        for action in plan["dataset_actions"]
    ]

    assert "remove_duplicates" in dataset_action_names

    # -------------------------
    # ID handling
    # -------------------------

    assert "user_id" in plan["column_actions"]

    user_id_actions = [
        action["action"]
        for action in plan["column_actions"]["user_id"]
    ]

    assert "drop_column" in user_id_actions

    # -------------------------
    # Empty / constant column
    # -------------------------

    assert "D" in plan["column_actions"]

    d_actions = [
        action["action"]
        for action in plan["column_actions"]["D"]
    ]

    assert "drop_column" in d_actions

    # -------------------------
    # Numeric preprocessing
    # -------------------------

    assert "A" in plan["column_actions"]

    a_actions = [
        action["action"]
        for action in plan["column_actions"]["A"]
    ]

    assert "median_imputation" in a_actions
    assert "standard_scaling" in a_actions

    # -------------------------
    # Datetime detection
    # -------------------------

    assert "start_date" in plan["column_actions"]

    start_date_actions = [
        action["action"]
        for action in plan["column_actions"]["start_date"]
    ]

    assert "extract_datetime_features" in start_date_actions
    assert "onehot_encoding" not in start_date_actions
    assert "label_encoding" not in start_date_actions