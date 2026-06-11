import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def dummy_df():
    """
    Creates a small messy DataFrame for fast unit testing.
    """
    data = {
        "A": [1, 2, np.nan, 4, 1], # Numeric with missing and duplicate
        "B": ["cat", "dog", "dog", np.nan, "cat"], # Categorical with missing
        "C": [10.5, np.nan, 30.1, 40.2, 10.5], # Float with missing
        "D": [None, None, None, None, None] # Completely missing
    }
    return pd.DataFrame(data)
