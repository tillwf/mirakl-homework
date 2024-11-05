# Mirakl

## Installation

Setup your virtual environment using `pyenv` ([pyenv installer](https://github.com/pyenv/pyenv-installer))

```bash
pyenv install 3.11.5
pyenv local 3.11.5
python -m venv venv
source venv/bin/activate
```

Then install the requirements and the package locally:


```
make install
```

## Setup

The files `data_train.csv.gz`, `data_test.csv.gz` and `category_parent.csv` must be in the folder `data/raw`

## Problem description


### Context


### Metric

- Business Metric / Online Metric



- ML Metric / Offline Metric



### Data


####  Label


#### Raw Features


## Commands

### Help

```bash
python -m mirakl_homework
```

Will display all possible commands with their description. You can display each command documentation with:

```bash
python -m mirakl_homework <command> --help
```

### Dataset Creation

Using the raw data we want to make a train/validation/test.
For the default values you can do:

```bash
make dataset
```

### Train the model

The neural network is implemented using Tensorflow to be able to visualize easily the training process using Tensorboard, to save and use the model quickly and to be able to complexify it without changing too much the code. It eases also the normalization of numerical features and the handling of categorical features as it will be embed in the graph.

```bash
make train
```

To check the training live, launch Tensorboard:

```
tensorboard --logdir logs
```

### Make predictions

Save the predictions and print the performance

```bash
make predictions
```

### Make tests

```bash
make tests
```

or

```bash
pytest tests
```

## Future Work


## Deployement


