from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import loras
import numpy as np
import pandas as pd

from constants import num_features, TARGET, seed


def loras_oversampling(original_Xtrain, original_ytrain):
    # LoRAS
    skf_df = original_Xtrain.copy()
    skf_df[TARGET] = original_ytrain
    # Separate the labels from features.
    # skf_df['Response'] = y
    skf_df_val = skf_df.values
    labels, features = skf_df_val[:, num_features], skf_df_val[:, :num_features]

    # Positive Response
    label_1 = np.where(labels == 1)[0]
    label_1 = list(label_1)
    print((len(label_1)))
    features_1 = features[label_1]
    features_1_train = features_1[list(range(0, round(len(label_1) / 2)))]  # od 0 do len(label_1)/2
    features_1_test = features_1[
        list(range(round(len(label_1) / 2), len(label_1)))]  # od len(label_1)/2 do len(label_1)

    # Negative Response
    label_0 = np.where(labels == 0)[0]
    label_0 = list(label_0)
    print(len(label_0))
    features_0 = features[label_0]
    features_0_train = features_0[list(range(0, round(len(label_0) / 2)))]
    features_0_test = features_0[list(range(round(len(label_0) / 2), len(label_0)))]

    training_data = np.concatenate((features_1_train, features_0_train))
    test_data = np.concatenate((features_1_test, features_0_test))
    training_labels = np.concatenate((np.zeros(round(len(label_1) / 2)) + 1, np.zeros(round(len(label_0) / 2))))
    test_labels = np.concatenate((np.zeros(round(len(label_1) / 2)) + 1, np.zeros(round(len(label_0) / 2))))

    min_class_points = features_1_train
    maj_class_points = features_0_train

    # LoRAS
    loras_min_class_points = loras.fit_resample(maj_class_points, min_class_points)
    print(loras_min_class_points.shape)
    LoRAS_feat = np.concatenate((loras_min_class_points, maj_class_points))
    LoRAS_labels = np.concatenate((np.zeros(len(loras_min_class_points)) + 1, np.zeros(len(maj_class_points))))
    return LoRAS_feat, LoRAS_labels


def smote_oversampling(X, y, strategy=None):
    if strategy is None:
        strategy = {0: 5000, 1: 5000}
    sm = SMOTE(random_state=seed, sampling_strategy=strategy)
    X_sm, y_sm = sm.fit_resample(X, y)
    return X_sm, y_sm


def random_undersampling(X, y, strategy=None):
    rus = RandomUnderSampler(random_state=seed)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res
