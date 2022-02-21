import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import scikitplot as skplt

from autoML.tpot import tpot_pipeline_optimizer
from clustering.clustering_model import Clustering
from constants import prep_cols, train_size, imputation_cols, scaling_cols, prep_cols_v2, training_file_path
from conversion_prediction.model import MLModel, NNModel
from preprocessing.data_scaling import standardize, normalize
from preprocessing.feature_engineering import feature_engineering
from preprocessing.imputation import Imputer
from preprocessing.outlier_removal import remove_outliers
from preprocessing.data_sampling import loras_oversampling
from validation.validation import print_response_rate, plot_fbeta_charts, \
    plot_pr_curve, shap_plot_summary, get_best_metrics, get_classification_report, plot_pr_curve_skplt
import yaml
import os


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def run_training_experiment():
    return


def load_and_preprocess(model_name=''):
    df = pd.read_csv('../data/marketing_campaign.csv', sep=';')
    # Response Rate
    print_response_rate(df)
    # Feature engineering
    df = feature_engineering(df)
    # Outlier removal
    df = remove_outliers(df)
    df.drop(['Education', 'Dt_Customer'], axis=1, inplace=True)

    # Imputation
    knn_imputer = Imputer(n_neighbors=5)
    knn_imputer.calculate_nans(df)
    df = knn_imputer.fit_transform(df, imputation_cols)
    # Dropna because the removed outliers left some NaNs
    df = df.dropna()

    # Cluster Preprocessing
    scaled_df = df.copy()
    scaled_df = standardize(scaled_df)
    scaled_df = normalize(scaled_df)
    # Clustering
    model = Clustering('kmeans')
    model.fit(scaled_df[scaling_cols])
    labels = model.predict(scaled_df[scaling_cols])
    df['Cluster'] = labels

    # Feature Preprocessing
    del scaled_df
    # df = standardize(df, prep_cols_v2)
    # df = normalize(df, prep_cols_v2)
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['Marital_Status', 'HasChild', 'Cluster'])

    y = df['Response']
    X = df.drop('Response', axis=1)
    # Train test split 80/20
    # Always split into test and train sets BEFORE trying oversampling techniques!
    # Oversampling before splitting the data can allow the exact same observations to be present
    # in both the test and train sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
    print("Train set: ", X_train.shape)
    print("Test set: ", X_test.shape)

    return X_train, y_train, X_test, y_test


def train_test_val_split(X,y):
    # 70/15/15 split
    # Always split into test and train sets BEFORE trying oversampling techniques!
    # Oversampling before splitting the data can allow the exact same observations to be present
    # in both the test and train sets.
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
    X_test, X_valid, y_test, y_valid = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42,
                                                        stratify=y_rem)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == "__main__":

    train_config = read_yaml(training_file_path)
    X_train, y_train, X_test, y_test = load_and_preprocess(train_config['model_name'])

    # transform the dataset using SMOTE
    if train_config['data_sampling'] == 'smote':
        sm = SMOTE(random_state=42)
        X_sm, y_sm = sm.fit_resample(X_train, y_train)
        X_train = X_sm
        y_train = y_sm
        print("Smote set: ", X_sm.shape)
    # transform the dataset using LoRAS
    if train_config['data_sampling'] == 'loras':
        X_loras, y_loras = loras_oversampling(X_train, y_train)
        print("Loras set: ", X_loras.shape)
        X_train = X_loras
        y_train = y_loras

    if train_config['use_tpot']:
        test_pipeline = tpot_pipeline_optimizer(X_train, y_train, X_test, y_test,
                                                generations=train_config['tpot_generations'],
                                                population_size=train_config['tpot_population'])
        # example pipeline
        # example_pipe = make_pipeline(
        #     StackingEstimator(estimator=XGBClassifier(learning_rate=0.5, max_depth=3, min_child_weight=8,
        #     n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)),
        #     StackingEstimator(estimator=XGBClassifier(learning_rate=0.5, max_depth=3, min_child_weight=8,
        #     n_estimators=100, n_jobs=1, subsample=1.0, verbosity=0)),
        #     StackingEstimator(estimator=GaussianNB()),
        #     BernoulliNB(alpha=10.0, fit_prior=False)
        # )
        # example_pipe.fit(X_train, y_train)
        y_probas = test_pipeline.predict_proba(X_test)

    # If not NN (Neural Network) Then use ML
    if not train_config['model_name'] == 'nn':
        # ML MODEL
        # TRAINING
        if train_config['cv']:
            clf = MLModel(train_config['model_name'])
            start = time.time()
            print('Starting Grid Search')
            grid_search = GridSearchCV(clf.model, param_grid=clf.param_grid, scoring='f1', n_jobs=-1, refit=True, cv=5)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            end = time.time()
            # summarize results
            print('Best CV params: ' + str(grid_search.best_params_))
            print('Best CV score: ' + str(grid_search.best_score_))
            print(f'GridSearchCV finished in: {end-start:0.2f} seconds')
        else:
            model = MLModel(train_config['model_name'])
            model.train(X_train, y_train)

        # VALIDATION
        predictions = model.predict(X_test)
        y_probas = model.predict_proba(X_test)
    else:
        # NEURAL NETWORK MODEL
        # TRAINING
        # scale data for NN
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        n_inputs = X_train.shape[1]

        model = NNModel()
        model.create_model(n_inputs=n_inputs)

        model.fit(X_train, y_train)

        # VALIDATION
        predictions = model.predict(X_test)
        y_probas = model.predict_proba(X_test)

    predictions = model.predict(X_test)
    y_probas_train = model.predict_proba(X_train)

    # plot_pr_curve(y_valid, y_probas, primary_beta=1)
    best_fscores, threshold_dict = plot_fbeta_charts(y_test, y_probas, model_name=model.name, timestamp=model.timestamp)
    metrics = get_best_metrics(y_test, y_probas, primary_beta=3, prefix='test')
    threshold = metrics['test_threshold']
    f1 = f1_score(y_test, predictions)
    print(f'regular f1: {f1}')
    print(metrics)
    print(get_classification_report(y_test, y_probas, threshold))
    print(best_fscores)
    shap_plot_summary(model, X_test, y_test, model.name, model.timestamp)
    plot_pr_curve_skplt(y_test, y_probas)


