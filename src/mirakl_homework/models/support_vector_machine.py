import joblib
import logging

from pathlib import Path
from sklearn import svm
from sklearn.metrics import accuracy_score

from .classifier import Classifier
from mirakl_homework.config import load_config

CONF = load_config()
C = CONF["svm"]["C"]
MODELS_ROOT = CONF["path"]["models_root"]
MODEL_PATH = f"{MODELS_ROOT}/svm.pkl"


# Support Vector Machine subclass
class SupportVectorMachine(Classifier):
    def __init__(self):
        self.C = C
        self.model = svm.LinearSVC(C=C)

    def fit(self, X_train, y_train, X_validation, y_validation, feature_cols):
        # Simulate fitting process
        logging.info(f"Training Support Vector Machine with kernel='linear' and C={self.C}.")
        self.model.fit(X_train[feature_cols], y_train)
        y_pred = self.model.predict(X_validation[feature_cols])
        logging.info(f"{accuracy_score(y_validation, y_pred)}:.2%")

    def predict(self, X_test, feature_cols):
        if not self.model:
            raise Exception("Model is not trained yet.")
        # Simulate prediction
        logging.info("Predicting with Support Vector Machine.")
        return self.model.predict(X_test[feature_cols])

    def save(self):
        Path(MODELS_ROOT).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)

    def load(self):
        self.model = joblib.load(MODEL_PATH)
