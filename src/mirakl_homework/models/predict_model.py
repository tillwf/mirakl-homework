import click
import logging
import os
import pickle

from pathlib import Path

from mirakl_homework.config import load_config
from mirakl_homework.utils import load_datasets
from mirakl_homework.models.utils import result_analysis

from .support_vector_machine import SupportVectorMachine
from .neural_network import NeuralNetwork

CONF = load_config()
MODEL_NAME = CONF["model"]["name"]
OUTPUT_ROOT = CONF["path"]["output_data_root"]


@click.group()
def predict():
    pass


@predict.command()
@click.option(
    '--model-name',
    type=str,
    default=MODEL_NAME,
    help='Model used for training, default is {}'.format(
        MODEL_NAME
    )
)
def make_predictions(model_name):
    logging.info("Make Prediction")

    logging.info("Reading test data")
    _, X_test, _ = load_datasets()
    y_test = X_test.pop("category_id")

    # Merge the "user based" features
    feature_cols = [f"f{i}" for i in range(128)]

    logging.info("Loading model")
    if model_name == "SVM":
        model = SupportVectorMachine()
        model.load()
    elif model_name == "NN":
        model = NeuralNetwork(feature_cols, y_test.nunique())
        model.load()
    else:
        print(f"No model named {model_name}")
        return

    raw_predictions = model.predict(X_test, feature_cols)

    # Add columns to compute the metrics (Mean Rank, MAP@10, etc.)
    X_test["predictions"] = raw_predictions
    result_analysis(y_test, raw_predictions)

    # Saving the predictions
    logging.info("Saving predictions")

    output_root = Path(OUTPUT_ROOT)
    with open(output_root / "category_mapping.pickle", 'rb') as file:
        uniques = pickle.load(file)
    X_test['final_predictions'] = X_test['predictions'].apply(
        lambda x: uniques[x]
    )
    X_test[["product_id", "final_predictions"]].to_csv(
        os.path.join(OUTPUT_ROOT, f"raw_predictions_{MODEL_NAME}.csv")
    )
