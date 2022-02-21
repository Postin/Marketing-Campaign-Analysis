import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, precision_recall_curve
import seaborn as sns
import numpy as np
import scikitplot as skplt

from constants import default_figsize


def plot_clusters(df, labels):
    df = df.values
    plt.scatter(df[:, 0], df[:, 1], c=labels, cmap='viridis')
    plt.show()


def get_silhouette_score(features, labels, model_name):
    print('{}: {}'.format(model_name, silhouette_score(features, labels, metric='cosine')))

def plot_correlation_heatmap(df):
    f, ax2 = plt.subplots(1, 1, figsize=(20, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 30}, ax=ax2)
    ax2.set_title('Correlation Matrix', fontsize=12)

def print_response_rate(df):
    print('No Responses', round(df['Response'].value_counts()[0] / len(df) * 100, 2), '% of the dataset')
    print('Responses', round(df['Response'].value_counts()[1] / len(df) * 100, 2), '% of the dataset')

def plot_shap():
    return

def plot_fbeta_charts(y_true, y_probas, primary_beta):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas)

    path = EVALUATION_PATH

def get_best_metrics(y_true, y_probas, primary_beta, prefix):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas)

    # fbeta formula
    f_scores = ((1 + np.power(primary_beta, 2)) * precision * recall) / (
                np.power(primary_beta, 2) * precision + recall)
    f_scores = f_scores[np.logical_not(np.isnan(f_scores))]

    # Identify index with max F score
    f_score_best_ix = np.argmax(f_scores)
    fscore_best = np.max(f_scores)
    best_precision = precision[f_score_best_ix]
    best_recall = recall[f_score_best_ix]
    best_threshold = thresholds[f_score_best_ix]

    metrics = {
        '{}_best_precision'.format(prefix): best_precision,
        '{}_best_recall'.format(prefix): best_recall,
        '{}_best_f{}'.format(prefix, primary_beta): fscore_best,
        '{}_threshold'.format(prefix): best_threshold,
        '{}_f{}_best_ix'.format(prefix, primary_beta): f_score_best_ix
    }

    return metrics


def plot_confusion_matrix(y_true, y_probas, threshold):
    y_pred = y_probas[:, 1] >= threshold
    skplt.metrics.plot_confusion_matrix(y_true, y_pred)


def plot_pr_curve(y_true, y_probas, primary_beta):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas)
    metrics = get_best_metrics(y_probas[:, 1], primary_beta)

    fscore_best_ix = metrics['None_f{}_best_ix'.format(primary_beta)]
    # Identify index with max F1 score
    plt.figure(figsize=default_figsize)
    plt.ylim(0, 1)

    plt.plot(recall, precision, marker='.')
    plt.scatter(recall[fscore_best_ix], precision[fscore_best_ix], marker='o', s=100, color='grey',
                label='Optimal (Best F{})'.format(primary_beta))
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.ylim(0, 1)