# Import required modules
import json
import pandas as pd
from pathlib import Path
from data_validation import check_data_schema, check_data_skew

# Read JSON
data_val_json_path = Path(__file__).resolve().parent / "config/data_validation.json"
with open(data_val_json_path, "r") as json_path:
    config_json = json.load(json_path)

# Get source data columns & data types
source_dtypes = config_json["source_data_types"]
source_columns = config_json["source_data_columns"]

# Read data for test
df = pd.read_csv("source_data/adult.csv")

def test_data_skew():
    missing_columns_list = check_data_skew(df, source_columns)
    assert len(missing_columns_list) == 0

def test_data_schema():
    dtype_diff_col_list = check_data_schema(df, source_dtypes)
    assert len(dtype_diff_col_list) == 0



