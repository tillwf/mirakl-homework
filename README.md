# Mirakl

Data Exploration and Experimentations in the [notebooks folder](notebooks/)

## Installation

To get started, set up your virtual environment using `pyenv` ([pyenv installer](https://github.com/pyenv/pyenv-installer))

```bash
pyenv install 3.11.5
pyenv local 3.11.5
python -m venv venv
source venv/bin/activate
```

Next, install the required dependencies and the package locally:

```
make install
```

## Setup

Ensure the following files are placed in the `data/raw` folder:

- `data_train.csv.gz`
- `data_test.csv.gz`
- `category_parent.csv`

## Problem description


### Context


The goal is to classify products into the leaf nodes of a category tree using a set of anonymous features (e.g., `f1`, `f2`, ..., `f127`). If a product is incorrectly placed in a non-leaf node or not assigned to any leaf node, it will be hidden from view, which may significantly disrupt the consumer experience.

### Metric

#### Business Metric / Online Metric

Our objective is to minimize both the number of misclassified articles and those that remain hidden.

To evaluate **visibility**, we calculate the percentage of products correctly assigned to leaf nodes. For **accuracy**, we monitor overall classification performance and accuracy per leaf node. This will help identify underperforming categories that need improvement.

#### ML Metric / Offline Metric

This is a standard multi-class classification problem, and **cross-entropy loss** is used as the primary objective function. For the neural network, we map non-consecutive category labels to consecutive integers during training and convert them back to original labels during export.

### Data Overview

- **Categories**: 101 categories in both train and test sets.
- **Articles**: 241,483 articles in the training set and 44,738 in the test set.
- Detailed data exploration can be found in the [notebooks](notebooks/) folder.


## Commands

### Help

To display all available commands with descriptions, run:

```bash
python -m mirakl_homework
```

For detailed documentation of a specific command, use:

```bash
python -m mirakl_homework <command> --help
```

### Dataset Creation

To generate the train/validation/test split and normalize the data using default settings, run:

```bash
make dataset
```

The default paths for data input and output are specified in the `config.yaml` file:

```yaml
path:
  input_data_path: "data/raw/"
  output_data_root: "data/processed/"
  interim_data_root: "data/interim/"
  models_root: "models"
  logs_root: "logs"
```

### Train the model

You can choose between two algorithm options for training: **SVM (Support Vector Machine)** or **Neural Network (NN)**.

- **SVM**: A primarily linear approach, but can be extended to non-linear problems using kernels. It is fast and efficient, especially for high-dimensional and small datasets.
- **Neural Network**: Built with TensorFlow, this model is ideal for capturing complex, non-linear relationships. TensorFlow integrates with **TensorBoard** for easy visualization of the training process, and it offers flexible model architecture adjustments.

To train the model, run:

```bash
make train
```

The model configuration (e.g., name, parameters) is specified in the `config.yaml` file:

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

If you're using the neural network, you can monitor the training process live with TensorBoard:

```
tensorboard --logdir logs
```

### Make predictions

To generate predictions and evaluate the model's performance, run:

```bash
make predictions
```

The predictions will be saved in `data/processed/raw_predictions.csv`.

### Make tests (not done)

To run tests on the codebase:


```bash
make tests
```

