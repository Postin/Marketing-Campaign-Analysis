from sklearn.impute import KNNImputer
import pandas as pd

from constants import imputation_cols


class Imputer:
    def __init__(self,  n_neighbors=5, metric='nan_euclidean' ):
        self.imputer = KNNImputer(n_neighbors=n_neighbors, metric=metric)

    def fit(self, df):
        self.imputer.fit(df)

    def transform(self, df, imputation_cols=imputation_cols):
        X = self.imputer.transform(df[imputation_cols])
        X = pd.DataFrame(X, columns=imputation_cols)
        df[imputation_cols] = X
        print('Transform complete. Missing values: {}'.format(df.isnull().values.sum()))
        return X

    def fit_transform(self, df, imputation_cols=imputation_cols):
        X = self.imputer.fit_transform(df[imputation_cols])
        # transform numpy array to Dataframe
        X = pd.DataFrame(X, columns=imputation_cols)
        df[imputation_cols] = X
        print('Fit transform complete. Missing values: {}'.format(df.isnull().values.sum()))
        return df

    def calculate_nans(self, df):
        print('Number of NaN values in dataframe: {}'.format(df.isnull().values.sum()))