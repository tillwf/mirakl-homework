import logging
import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from .classifier import Classifier
from mirakl_homework.config import load_config

CONF = load_config()
EPSILON = tf.keras.backend.epsilon()
EPOCH = CONF["nn"]["epoch"]
PATIENCE = CONF["nn"]["early_stopping_patience"]

MODELS_ROOT = CONF["path"]["models_root"]
LOGS_ROOT = CONF["path"]["logs_root"]


class NeuralNetwork(Classifier):
    def __init__(self, feature_cols, n_categories):
        self.feature_cols = feature_cols
        self.n_categories = n_categories

        logging.info("Training Neural Network with 3 layers.")
        self.model = self.build_model()
        self.callbacks = self.create_callbacks()
        self.model.summary()

    def build_model(self):
        l2_weight = 10 ** -5

        inputs = Input(shape=(len(self.feature_cols),), name="feature_input")
        x = Dropout(0.2)(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(
            self.n_categories,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
            name="output_layer"
        )(x)

        model = Model(inputs=inputs, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        return model

    def create_callbacks(self):
        # Add callbacks to be able to restart if a process fail, to
        # save the best model and to create a TensorBoard
        os.makedirs(MODELS_ROOT, exist_ok=True)
        os.makedirs(LOGS_ROOT, exist_ok=True)

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=LOGS_ROOT,
            histogram_freq=1,
            update_freq='epoch'
        )
        best_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_ROOT, "best_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True
        )

        return [tensorboard, best_model_checkpoint, early_stopping]

    def fit(self, X_train, y_train, X_validation, y_validation, feature_cols):
        # Launch the train and save the loss evolution in `history`
        history = self.model.fit(
            X_train[feature_cols],
            y_train.values,
            callbacks=self.callbacks,
            epochs=EPOCH,
            validation_data=(
                X_validation[feature_cols],
                y_validation.values
            )
        )
        return history

    def predict(self, X_test, feature_cols):
        if not self.model:
            raise Exception("Model is not trained yet.")

        logging.info("Predicting with Neural Network.")
        raw_predictions = self.model.predict(X_test[feature_cols])
        return np.argmax(raw_predictions, axis=1)

    def save(self, filename="final_model.h5"):
        save_path = os.path.join(MODELS_ROOT, filename)
        self.model.save(save_path)
        logging.info(f"Model saved to {save_path}")

    def load(self, filename="final_model.h5"):
        load_path = os.path.join(MODELS_ROOT, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file {load_path} not found.")
        self.model = load_model(load_path)
        logging.info(f"Model loaded from {load_path}")
        return self.model
