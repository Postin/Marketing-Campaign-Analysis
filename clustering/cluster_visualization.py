import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cluster_3d_scatterplot(df):
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)

    x = df['Income']
    y = df['Spending']
    z = df['Seniority']
    c = df['Cluster']

    ax.set_xlabel("Income")
    ax.set_ylabel("Spending")
    ax.set_zlabel("Seniority")

    ax.scatter(xs=x, ys=y, zs=z, c=c)
    plt.show()