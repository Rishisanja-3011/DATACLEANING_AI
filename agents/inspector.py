import pandas as pd

def inspect_data(df):

    report = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "columns": df.dtypes.astype(str).to_dict(),
        "unique_values": df.nunique().to_dict()
    }
    return report


