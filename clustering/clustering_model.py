import json

from constants import seed, MODEL_PATH
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

class Clustering:
    def __init__(self, model_name):
        self.model = ModelsFactory.from_name(model_name)
        self.name = model_name

    def fit(self, df):
        self.model.fit(df)

    def predict(self, df):
        return self.model.predict(df)


class ModelsFactory:
    """
    Factory class that instantiates a model.
    """
    @staticmethod
    def from_name(model_name):
        params = json.load(open(MODEL_PATH.format(model_name), 'r'))
        if model_name.lower() == 'kmeans':
            return KMeans(random_state=seed, **params)
        elif model_name.lower() == 'dbscan':
            return DBSCAN(**params)
        elif model_name.lower() == 'gmm':
            return GaussianMixture(random_state=seed, **params)