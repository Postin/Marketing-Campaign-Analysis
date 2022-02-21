from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def elbow_method(n_clusters, df):
    distortions = []
    for k in range(1, n_clusters):
        kmeanModel = KMeans(n_clusters=k).fit(df)
        distortions.append(sum(np.min(cdist(df,
                                            kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])

    # Plot the elbow
    K = [*range(1, n_clusters)]
    plt.plot(K, distortions, 'rx-')
    plt.xlabel('K')
    plt.ylabel('Suma kvadrata greške')
    plt.title('Metoda lakta za računanje optimalnog K')
    plt.show()