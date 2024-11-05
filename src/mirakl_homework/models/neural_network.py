import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from .classifier import Classifier
from mirakl_homework.config import load_config
from mirakl_homework.models.utils import dataframe_to_dict

CONF = load_config()
EPSILON = tf.keras.backend.epsilon()
EPOCH = CONF["nn"]["epoch"]
PATIENCE = CONF["nn"]["early_stopping_patience"]

MODELS_ROOT = CONF["path"]["models_root"]
LOGS_ROOT = CONF["path"]["logs_root"]


class NeuralNetwork(Classifier):
    def __init__(self, feature_cols, n_categories):
        # Simulate fitting process
        print("Training Neural Network with 3 layers.")
        # Replace with actual neural network training code

        inputs = []
        encoded_features = []

        l2_weight = 10 ** -5

        # Normalize numerical features
        for col in feature_cols:
            numeric_input = Input(shape=(1,), name=col)
            normalization_layer = Normalization()(numeric_input)
            inputs.append(numeric_input)
            encoded_features.append(normalization_layer)

        all_features = Concatenate()(encoded_features)

        # Define the rest of the model
        x = Dense(64, activation='relu')(all_features)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(
            n_categories,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
            name="category"
        )(x)

        # Create the model
        self.model = Model(inputs=inputs, outputs=output)

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Model summary
        self.model.summary()

        # Add callbacks to be able to restart if a process fail, to
        # save the best model and to create a TensorBoard
        self.callbacks = []

        os.makedirs(MODELS_ROOT, exist_ok=True)
        os.makedirs(LOGS_ROOT, exist_ok=True)
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=LOGS_ROOT,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq=100,
            profile_batch=2,
            embeddings_freq=1
        )
        self.callbacks.append(tensorboard)

        self.best_model_file = os.path.join(MODELS_ROOT, "best_model_so_far")
        best_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.best_model_file,
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )
        self.callbacks.append(best_model_checkpoint)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=PATIENCE,
            monitor="val_loss"
        )
        self.callbacks.append(early_stopping)

    def fit(self, X_train, y_train, X_validation, y_validation, feature_cols):

        # Prepare data for training (convert X_train to dictionary)
        X_train_dict = dataframe_to_dict(X_train, feature_cols)
        X_validation_dict = dataframe_to_dict(X_validation, feature_cols)

        # Launch the train and save the loss evolution in `history`
        history = self.model.fit(
            X_train_dict,
            y_train.values,
            callbacks=self.callbacks,
            epochs=EPOCH,
            validation_data=(
                X_validation_dict,
                y_validation.values
            )
        )

        # Save the model
        self.model.load_weights(self.best_model_file)

    def predict(self, X_test, feature_cols):
        if not self.model:
            raise Exception("Model is not trained yet.")
        # Simulate prediction
        print("Predicting with Neural Network.")
        X_test_dict = dataframe_to_dict(X_test, feature_cols)

        print("Making predictions")
        raw_predictions = self.model.predict(X_test_dict)
        return np.argmax(raw_predictions, axis=1)

    def save(self):
        self.model.save(os.path.join(MODELS_ROOT, "final_model"))

    def load(self):
        model = load_model(
            os.path.join(MODELS_ROOT, "final_model")
        )
        return model
