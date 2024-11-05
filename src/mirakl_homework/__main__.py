import click
from mirakl_homework.data.make_dataset import dataset
from mirakl_homework.models.train_model import train
from mirakl_homework.models.predict_model import predict

cli = click.CommandCollection(sources=[
    dataset,
    predict,
    train,
])

if __name__ == '__main__':
    cli()
