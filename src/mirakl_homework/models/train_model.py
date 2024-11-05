import click
import logging

from mirakl_homework.config import load_config
from mirakl_homework.utils import load_datasets
from mirakl_homework.models.neural_network import NeuralNetwork
from mirakl_homework.models.support_vector_machine import SupportVectorMachine

CONF = load_config()
MODEL_NAME = CONF["model"]["name"]


@click.group()
def train():
    pass


@train.command()
@click.option(
    '--model-name',
    type=str,
    default=MODEL_NAME,
    help='Model used for training, default is {}'.format(
        MODEL_NAME
    )
)
def train_model(model_name):
    logging.info("Training Model")
    X_train, X_validation, _ = load_datasets()

    y_train = X_train.pop("category_id")
    y_validation = X_validation.pop("category_id")

    # Select feature columns
    feature_cols = [f"f{i}" for i in range(128)]

    if model_name == "NN":
        print("NN")
        model = NeuralNetwork(feature_cols, y_train.nunique())
    elif model_name == "SVM":
        print("SVM")
        model = SupportVectorMachine()
    else:
        print("Model not implemented")
        return

    model.fit(X_train, y_train, X_validation, y_validation, feature_cols)
    model.save()
