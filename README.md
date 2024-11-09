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

The objective is to accurately classify products into the leaf nodes of a category tree using a set of anonymous features (f1, ..., f127). If a product is not placed in a leaf node, it remains hidden from view, and if it is assigned to an incorrect leaf, it can significantly disrupt the consumer experience.

### Metric

- Business Metric / Online Metric

Our aim is to minimize both the number of misclassified articles and the number of articles that remain hidden.

To measure visibility, we can calculate the percentage of products assigned to leaf nodes. For accuracy, we can monitor overall classification accuracy, as well as accuracy per leaf node, to identify and address the weakest-performing nodes.


- ML Metric / Offline Metric

As the problem is a traditional classification problem, the loss chosen is the cross entropy. In order to train the neural network we will map non-consecutive labels to consecutive integers and convert back during the export.

### Data

The are 101 categories in train and test. 241483 articles in train and 44738 in test. The full exploration of data can be found in the [notebooks](notebooks/) folder.

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

Using the raw data we want to make a train/validation/test and normalize the data.
For the default values you can do:

```bash
make dataset
```

The default path are in the config file `config.yaml`:

```yaml
path:
  input_data_path: "data/raw/"
  output_data_root: "data/processed/"
  interim_data_root: "data/interim/"
  models_root: "models"
  logs_root: "logs"
```

### Train the model

You have two algorithm options: SVM (Support Vector Machine) for a primarily linear approach (though it can be extended to non-linear problems with kernels), and NN (Neural Network) for capturing complex, non-linear relationships between features and labels.

The neural network is implemented using TensorFlow, providing several advantages. TensorFlow's integration with TensorBoard enables easy visualization of the training process, while its architecture allows for quick model saving and deployment. Additionally, TensorFlow simplifies model complexity adjustments with minimal code changes. It also streamlines normalization of numerical features and embedding of categorical features directly within the computation graph, making feature handling more efficient.

SVM are implemented using Scikit Learn ([LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)). It is a fast and efficient algorithm, especially suited to high-dimensional data and smaller datasets.

```bash
make train
```

The model name and its parameters are in the config file `config.yaml`:

```yaml
model:
  name: "NN"

nn:
  version: 1
  epoch: 50
  early_stopping_patience: 5

svm:
  C: 1.0
```

If you choose the neural network you can check the training live using Tensorboard:

```
tensorboard --logdir logs
```

### Make predictions

Save the predictions and print the performance

```bash
make predictions
```

The prediction will be saved in the folder `data/processed/raw_predictions.csv`

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


