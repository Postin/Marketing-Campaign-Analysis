import time

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, precision_recall_curve, confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import scikitplot as skplt
from matplotlib.backends.backend_pdf import PdfPages
from constants import default_figsize, EVALUATION_PATH, shap_plot_size
import shap

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


def plot_fbeta_charts(y_true, y_probas, model_name, timestamp, n=10):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas[:, 1])

    path = EVALUATION_PATH + '/' + model_name + '_' + timestamp + '_fbeta_charts.pdf'
    pp = PdfPages(path)

    tps = []
    fps = []
    best_fscores = {}
    annoatations = []
    temp = 0
    thresholds_dict = {}
    for i in range(1, n+1):
        # fbeta formula
        f_scores = ((1 + np.power(i, 2)) * precision * recall) / (
                np.power(i, 2) * precision + recall)
        f_scores = f_scores[np.logical_not(np.isnan(f_scores))]

        fscore_best_ix = np.argmax(f_scores)
        fscore_best = np.max(f_scores)
        best_fscores['f{}'.format(i)] = fscore_best
        best_threshold = thresholds[fscore_best_ix]
        thresholds_dict['f{}'.format(i)] = thresholds[fscore_best_ix]
        plt.clf()
        plot_pr_curve(y_true, y_probas, i)
        pp.savefig()
        plt.clf()
        plot_confusion_matrix(y_true, y_probas, best_threshold)
        pp.savefig()

        # Scatter plot for fbeta 1 through n
        # Calculate TP, FP
        y_pred = y_probas[:, 1] >= best_threshold
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # If the threshold changed, so the number of FP changed
        if fp != temp:
            fps.append(fp)
            tps.append(tp)
            temp = fp
            annoatations.append('f{}'.format(i))

    # Make scatter plot
    plt.figure(figsize=default_figsize)
    tps_fps = [a + b for a, b in zip(tps, fps)]
    plt.scatter(tps_fps, tps)
    plt.title('Fbeta scatter plot')
    plt.ylabel('TP')
    plt.xlabel('TP+FP')
    for i, label in enumerate(annoatations):
        plt.annotate(label, (tps_fps[i], tps[i]))
    pp.savefig()
    pp.close()

    return best_fscores, thresholds_dict


def plot_pr_curve_skplt(y_true, y_probas):
    skplt.metrics.plot_precision_recall(y_true, y_probas, plot_micro=False, classes_to_plot=1)
    plt.show()


def get_best_metrics(y_true, y_probas, primary_beta, prefix):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas[:, 1])

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

def get_classification_report(y_true, y_probas, threshold):
    y_pred = y_probas[:, 1] >= threshold
    report = classification_report(y_true, y_pred)
    return report

def plot_confusion_matrix(y_true, y_probas, threshold):
    y_pred = y_probas[:, 1] >= threshold
    skplt.metrics.plot_confusion_matrix(y_true, y_pred)


def plot_pr_curve(y_true, y_probas, primary_beta):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas[:, 1])
    metrics = get_best_metrics(y_true, y_probas, primary_beta, prefix='None')

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

def shap_plot_summary(model, X_valid, y_valid, model_name, timestamp):
    explainer = model.get_explainer(X_valid)

    path = EVALUATION_PATH + '/' + model_name + '_' + timestamp + '_SHAP.pdf'
    pp = PdfPages(path)

    if isinstance(explainer, shap.explainers._tree.Tree):
        shap_values = explainer.shap_values(X_valid, y_valid)
    else:
        shap_values = explainer.shap_values(X_valid)

    plt.clf()
    shap.summary_plot(shap_values, X_valid, plot_type='dot', show=False, plot_size=shap_plot_size)
    pp.savefig()
    pp.close()
    plt.clf()