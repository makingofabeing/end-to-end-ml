"""
The data is assumed to be available in disk. Further, let's assume the data available in disk as
new data to train the model on.
"""

# Import required dependencies
import sys
import json
from pathlib import Path
import pandas as pd


def check_data_schema(df, base_dtype_dict):
    df = df.drop("income", axis=1)

    df_dtypes = pd.DataFrame(df.dtypes).reset_index(drop=False)
    df_dtypes = df_dtypes.rename(columns={"index": "source_columns", 0: "data_type"})
    df_dtypes["data_type"] = df_dtypes["data_type"].astype("string")
    source_dtype_dict = dict(zip(df_dtypes["source_columns"], df_dtypes["data_type"]))

    dtype_diff_col_list = []

    if base_dtype_dict == source_dtype_dict:
        print("Data type of source columns remain intact")
    else:

        for key in base_dtype_dict.keys():
            if base_dtype_dict[key] != source_dtype_dict[key]:
                dtype_diff_col_list.append(key)

        print(f"Data type of source column has changed: {dtype_diff_col_list}")

    return dtype_diff_col_list


def check_data_skew(df, source_data_columns):
    missing_cols = []

    for column in source_data_columns:
        if column not in df.columns:
            missing_cols.append(column)

    if len(missing_cols) > 0:
        print(f"Skew detected! Missing columns: {missing_cols}")
    else:
        print("No data skew detected")

    return missing_cols


if __name__ == "__main__":
    data_val_json_path = Path(__file__).resolve().parent / "config/data_validation.json"
    with open(data_val_json_path, "r") as json_path:
        config_json = json.load(json_path)

    # Get source data columns & data types
    source_dtypes = config_json["source_data_types"]
    source_columns = config_json["source_data_columns"]

    # Read data
    filepath = sys.argv[-1]
    df = pd.read_csv(filepath)

    # Validate data schema
    check_data_schema(df, source_dtypes)

    # Validate data skew
    check_data_skew(df, source_columns)