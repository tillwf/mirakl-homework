from abc import ABC, abstractmethod

# Base Classifier class with abstract methods
class Classifier(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_validation, y_validation, feature_cols):
        pass

    @abstractmethod
    def predict(self, X_test, feature_cols):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
