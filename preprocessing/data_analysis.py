import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize': (5, 6)})

def numeric_data_analysis(df):
    cols = list(df.select_dtypes(include=[np.number]).columns.values)
    for col in cols:
        print('{} min: {}, max: {}, mean: {}'.format(col, df[col].min(), df[col].max(), df[col].mean()))


def categorical_data_analysis(df):
    cols = list(df.select_dtypes(include=[object]).columns.values)
    print(cols)

def plot_boxplots(df):
    cols = list(df.select_dtypes(include=[np.number]).columns.values)
    f, axes = plt.subplots(nrows=1, ncols=3)

    # sns.boxplot(y="Age", data=df, ax=axes[0])
    # axes[0].set_title("Age boxplot")
    # #
    # sns.boxplot(y="Income", data=df, ax=axes[1])
    # axes[1].set_title("Income boxplot")
    # #
    # sns.boxplot(y="Spending", data=df, ax=axes[2])
    # axes[2].set_title("Spending boxplot")

    sns.boxplot(y="MntWines", data=df, ax=axes[0])
    axes[0].set_title("Wine spending boxplot")
    #
    sns.boxplot(y="MntFruits", data=df, ax=axes[1])
    axes[1].set_title("Fruit spending boxplot")
    #
    sns.boxplot(y="MntMeatProducts", data=df, ax=axes[2])
    axes[2].set_title("Meat spending boxplot")

    plt.show()
    #
    # sns.boxplot(y="Educational_Years", data=df, ax=axes[3])
    # axes[3].set_title("Educational Years boxplot")
    #
    # sns.boxplot(y="Seniority", data=df, ax=axes[4])
    # axes[4].set_title("Seniority boxplot")


def plot_boxplotsv1(df):
    cols = list(df.select_dtypes(include=[np.number]).columns.values)

    n_cols = 3
    n_rows = math.floor(len(cols) / 3)
    y = 0
    f, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            sns.boxplot(y=cols[y], data=df, ax=axes[i, j])
            axes[i, j].set_title('{} boxplot'.format(cols[y]))
            y = y + 1
        plt.show()



    # sns.boxplot(y="Age", data=df, ax=axes[0])
    # axes[0].set_title("Age boxplot")
    #
    # sns.boxplot(y="Income", data=df, ax=axes[1])
    # axes[1].set_title("Income boxplot")
    #
    # sns.boxplot(y="Spending", data=df, ax=axes[2])
    # axes[2].set_title("Spending boxplot")
    #
    # sns.boxplot(y="Educational_Years", data=df, ax=axes[3])
    # axes[3].set_title("Educational Years boxplot")
    #
    # sns.boxplot(y="Seniority", data=df, ax=axes[4])
    # axes[4].set_title("Seniority boxplot")


