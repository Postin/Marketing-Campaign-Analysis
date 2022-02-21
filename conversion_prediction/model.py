import json

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from constants import seed, MODEL_PATH
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


class MLModel:
    def __init__(self, model_name):
        self.model = ModelsFactory.from_name(model_name)
        self.name = model_name

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, input):
        return self.model.predict(input)

    def predict_proba(self, input):
        return self.model.predict_proba(input)


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