import pandas as pd

from clustering.cluster_analysis import cluster_mean
from clustering.cluster_visualization import cluster_3d_scatterplot
from clustering.clustering_model import Clustering
from clustering.elbow_method import elbow_method
from constants import imputation_cols, scaling_cols
from preprocessing.data_analysis import numeric_data_analysis, categorical_data_analysis, plot_boxplots
from validation.validation import plot_clusters, get_silhouette_score
from preprocessing.data_scaling import standardize, normalize
from preprocessing.feature_engineering import feature_engineering, make_cluster, make_bin_df, print_bin_explainer
from preprocessing.imputation import Imputer
from preprocessing.outlier_removal import remove_outliers
from rule_mining.apriori import Apriori

pd.options.mode.chained_assignment = None
#Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 999)
pd.options.display.float_format = "{:.3f}".format


if __name__ == "__main__":
    # Read data
    df = pd.read_csv('data/marketing_campaign.csv', sep=';')
    # Data analysis
    numeric_data_analysis(df)
    categorical_data_analysis(df)

    # Feature engineering
    df = feature_engineering(df)
    print(df[['Age','Income','Spending']].describe())
    # plot_boxplots(df)
    # Remove Outliers
    df = remove_outliers(df)


    # Imputation
    knn_imputer = Imputer()
    knn_imputer.calculate_nans(df)
    df = knn_imputer.fit_transform(df, imputation_cols)
    # Dropna because the removed outliers left some NaNs
    df = df.dropna()

    # Standardization
    scaled_df = df.copy()
    # Data scaling must be done before distance based algorithms like Clustering, SVM, PCA, KNN
    scaled_df = standardize(scaled_df)
    # Normalization
    scaled_df = normalize(scaled_df)
    # elbow_method(n_clusters=10, df=df[scaling_cols])

    # Clustering
    model = Clustering('gmm')
    model.fit(scaled_df[scaling_cols])
    labels = model.predict(scaled_df[scaling_cols])
    # plot_clusters(scaled_df[scaling_cols], labels)
    # print(get_silhouette_score(scaled_df[scaling_cols], labels, 'kmeans'))

    # Cluster interpretation and analysis
    df['Cluster'] = labels
    # cluster_3d_scatterplot(df)
    cluster_mean(df)
    df = make_cluster(df)

    df_bin = make_bin_df(df)
    print_bin_explainer(df)
    df_assoc = pd.get_dummies(df_bin)
    analyzer = Apriori(df_assoc)
    wine_results = analyzer.analyze(product='WineGroup', segment='Biggest consumer')
    fruit_results = analyzer.analyze(product='FruitGroup', segment='Biggest consumer')
    meat_results = analyzer.analyze(product='MeatGroup', segment='Biggest consumer')
    sweets_results = analyzer.analyze(product='SweetGroup', segment='Biggest consumer')
    fish_results = analyzer.analyze(product='FishGroup', segment='Biggest consumer')
    print(sweets_results.head(5))
