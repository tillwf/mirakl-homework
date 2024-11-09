import click
import logging
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from pathlib import Path

from mirakl_homework.config import load_config
from mirakl_homework.utils import load_data

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["output_data_root"]
FEATURES = [f"f{i}" for i in range(127)]


@click.group()
def dataset():
    pass


@dataset.command()
@click.option(
    '--train-eval-ratio',
    type=float,
    default=0.2,
    help='Split train/evaluation ratio. Default is 0.2'
)
def make_dataset(train_eval_ratio):
    logging.info("Making Dataset")
    logging.info(f"Loading from {DATA_PATH}")
    logging.info(f"Outputting to {OUTPUT_ROOT}")

    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("Loading raw data")
    df_train, df_test = load_data()

    # Replace category id by continuous values
    codes, uniques = pd.factorize(list(df_train['category_id'].values))
    df_train["category_id"] = codes
    df_test["category_id"] = df_test["category_id"].apply(lambda x: list(uniques).index(x))

    logging.info("Splitting train into train/eval")
    df_train, df_eval = train_test_split(
        df_train,
        test_size=train_eval_ratio,
        stratify=df_train["category_id"],
        random_state=84
    )

    logging.info("Normalizing features")
    train_means = df_train[FEATURES].mean()
    train_stds = df_train[FEATURES].std()

    df_train[FEATURES] = (df_train[FEATURES] - train_means) / train_stds
    df_eval[FEATURES] = (df_eval[FEATURES] - train_means) / train_stds
    df_test[FEATURES] = (df_test[FEATURES] - train_means) / train_stds

    logging.info("Saving Files")

    # TRAINSET
    logging.info("\tTrainset")
    train_path = output_root / "train.parquet"
    df_train.to_parquet(train_path, index=False)

    # VALIDATION SET
    logging.info("\tValidation set")
    eval_path = output_root / "eval.parquet"
    df_eval.to_parquet(eval_path, index=False)

    # TESTSET
    logging.info("\tTestset")
    test_path = output_root / "test.parquet"
    df_test.to_parquet(test_path, index=False)
    
    # Mapping
    logging.info("\tMapping")
    with open(output_root / "category_mapping.pickle", 'wb') as file:
        pickle.dump(uniques, file)

    # Sanity Check
    train_users = pd.read_parquet(train_path)
    logging.info(f"train size: {len(train_users)} ({train_users.category_id.nunique()} distinct category)")

    eval_users = pd.read_parquet(eval_path)
    logging.info(f"eval size: {len(eval_users)} ({eval_users.category_id.nunique()} distinct category)")

    test_users = pd.read_parquet(test_path)
    logging.info(f"test size: {len(test_users)} ({test_users.category_id.nunique()} distinct category)")
