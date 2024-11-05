.PHONY: clean install dataset train predictions tests

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = mirakl_homework
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
install:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Make Dataset
dataset:
	$(PYTHON_INTERPRETER) -m mirakl_homework make-dataset

train:
	$(PYTHON_INTERPRETER) -m mirakl_homework train-model

predictions:
	$(PYTHON_INTERPRETER) -m mirakl_homework make-predictions

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm data/processed/*.parquet
	rm -r models/*

## Test python environment is setup correctly
tests:
	$(PYTHON_INTERPRETER) -m pytest tests
