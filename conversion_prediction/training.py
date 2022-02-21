import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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
    plot_pr_curve, shap_plot_summary
import yaml
import os


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def run_training_experiment():
    return


if __name__ == "__main__":
    df = pd.read_csv('../data/marketing_campaign.csv', sep=';')

    train_config = read_yaml(training_file_path)

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
    df = standardize(df, prep_cols_v2)
    df = normalize(df, prep_cols_v2)
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['Marital_Status', 'HasChild', 'Cluster'])

    y = df['Response']
    X = df.drop('Response', axis=1)

    # Always split into test and train sets BEFORE trying oversampling techniques!
    # Oversampling before splitting the data can allow the exact same observations to be present
    # in both the test and train sets.
    # 70/15/15 split
    if train_config['cv']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
        print("Train set: ", X_train.shape)
        print("Test set: ", X_test.shape)
    else:
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42,
                                                            stratify=y_rem)
        print("Train set: ", X_train.shape)
        print("Validation set: ", X_valid.shape)
        print("Test set: ", X_test.shape)



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
        test_pipeline = tpot_pipeline_optimizer(X_train, y_train, X_valid, y_valid)
        y_probas = test_pipeline.predict_proba(X_valid)
        y_probas = y_probas[:, 1]
        # example pipeline
        example_pipe = make_pipeline(
            StandardScaler(),
            MLPClassifier(alpha=0.01, learning_rate_init=0.01)
        )
        example_pipe.fit(X_train, y_train)

    # If not NN (Neural Network) Then use ML
    if not train_config['model_name'] == 'nn':
        # ML MODEL
        # TRAINING
        if train_config['cv']:
            clf = MLModel(train_config['model_name'])
            start = time.time()
            grid_search = GridSearchCV(clf, param_grid=clf.param_grid, scoring = 'AUC', n_jobs=-1, refit=True, cv=5)
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
        predictions = model.predict(X_valid)
        y_probas = model.predict_proba(X_valid)
    else:
        # NEURAL NETWORK MODEL
        # TRAINING
        n_inputs = X_train.shape[1]
        model = NNModel()
        model.create_model(n_inputs=n_inputs)

        model.fit(X_train, y_train)

        # VALIDATION
        predictions = model.predict(X_valid)
        y_probas = model.predict_proba(X_valid)

    plot_pr_curve(y_valid, y_probas, primary_beta=5)
    shap_plot_summary(model, X_valid, y_valid, model.name, model.timestamp)
    best_fscores, threshold_dict = plot_fbeta_charts(y_valid, y_probas, model.name, model.timestamp)
    print(classification_report(y_valid, predictions))
    print(best_fscores)


