import json
import time

from catboost import CatBoostClassifier
import keras
from keras import Sequential, regularizers
from keras.layers import Dense, Dropout
from keras.optimizer_v2.nadam import Nadam
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from constants import seed, MODEL_PATH, PARAM_GRID_PATH
import shap

class MLModel:
    def __init__(self, model_name):
        self.model = ModelsFactory.from_name(model_name)
        self.name = model_name
        self.param_grid = self.get_param_grid(self.name)

        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
        self.timestamp = timestamp

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, input):
        return self.model.predict(input)

    def predict_proba(self, input):
        return self.model.predict_proba(input)

    def get_explainer(self, X):
        if self.name.lower() == 'xgb':
            return shap.TreeExplainer(self.model)
        elif self.name.lower() == 'catboost':
            return shap.TreeExplainer(self.model)
        elif self.name.lower() == 'rf':
            return shap.KernelExplainer(self.model.predict, X)

    def get_param_grid(self, model_name):
        param_grid =json.load(open(PARAM_GRID_PATH.format(model_name), 'r'))
        return param_grid


class ModelsFactory:
    """
    Factory class that instantiates a model.
    """
    @staticmethod
    def from_name(model_name):
        params = json.load(open(MODEL_PATH.format(model_name), 'r'))
        if model_name.lower() == 'xgb':
            return XGBClassifier(random_state=seed, **params)
        elif model_name.lower() == 'catboost':
            return CatBoostClassifier(**params)
        elif model_name.lower() == 'rf':
            return RandomForestClassifier(random_state=seed, **params)


class NNModel:
    def __init__(self):
        self.name = 'Neural Network Model'

        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
        self.timestamp = timestamp

    def create_model(self, n_inputs, activation='softsign', init_mode='lecun_uniform', num_layers=1, l2_penalty=0.001,
                     dropout=0.4, num_neurons=5, learning_rate=0.0001, loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy']):
        layers = []
        layers.append(Dense(n_inputs, input_shape=(n_inputs,), activation='relu', kernel_initializer=init_mode,
                            kernel_regularizer=regularizers.l2(l2_penalty)))
        for i in range(0, num_layers):
            layers.append(Dense(num_neurons, activation=activation, kernel_initializer=init_mode,
                                kernel_regularizer=regularizers.l2(l2_penalty)))
            layers.append(Dropout(dropout))
        layers.append(Dense(num_neurons, activation=activation, kernel_initializer=init_mode,
                            kernel_regularizer=regularizers.l2(l2_penalty)))
        layers.append(Dense(2, activation='softmax'))

        self.model = Sequential(layers)
        self.model.compile(Nadam(lr=learning_rate), loss=loss, metrics=metrics)

    def fit(self, X_train, y_train, batch_size=8, epochs=300, callbacks=[]):
        start = time.time()
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_split=0.3)
        end = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(end-start))
        print(f"NN Training finished in {end - start}")

    def load_model(self, model_path=''):
        self.model = keras.models.load_model(model_path)

    def predict_proba(self, X):
        y_probas = self.model.predict(X)
        return y_probas

    def predict(self, X):
        predictions = np.argmax(self.model.predict(X), axis=-1)
        return predictions
