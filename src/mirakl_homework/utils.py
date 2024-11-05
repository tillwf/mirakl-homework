import os
import pandas as pd

from pathlib import Path

from mirakl_homework.config import load_config

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
TRAIN_FILENAME = "data_train.csv.gz"
TEST_FILENAME = "data_test.csv.gz"
OUTPUT_ROOT = CONF["path"]["output_data_root"]


def load_data():
    data_path = Path(DATA_PATH)
    return (
        pd.read_csv(filepath_or_buffer=data_path / TRAIN_FILENAME),
        pd.read_csv(filepath_or_buffer=data_path / TEST_FILENAME)
    )


def load_datasets():
    return (
        pd.read_parquet(os.path.join(OUTPUT_ROOT, "train.parquet")),
        pd.read_parquet(os.path.join(OUTPUT_ROOT, "eval.parquet")),
        pd.read_parquet(os.path.join(OUTPUT_ROOT, "test.parquet")),
    )
