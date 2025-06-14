import numpy as np
from sklearn.datasets import  load_wine
from sklearn.cluster import  KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
# TruncatedSVD is used mostly for sparse data
# could be replaced with PCA
import matplotlib.pyplot as plt

data = load_wine()
svd_estimator = TruncatedSVD().fit(data.data)
points_2d = svd_estimator.transform(data.data)

sums_of_squares_for_differen_n = list()
for n in range(2, 10):
    clustering = KMeans(n)
    estimator = clustering.fit(data.data)
    preds = estimator.predict(data.data)
    arr_plot = np.concatenate([preds.reshape(-1,1), points_2d], axis=1)
    # matplotlib plot the points at arr_plot where the first dimension represents the color and the second and third dimensions are the x and y coordinates
    # plt.scatter(arr_plot[:,1], arr_plot[:,2], c=arr_plot[:,0], cmap='viridis', s=10)
    # plt.show()

    # Calculate the sum of squared distances
    inertia = estimator.inertia_
    sums = list()
    for i in range(n):
        obs = data.data[preds == i]
        centroids_point = np.stack([estimator.cluster_centers_[i] for _ in obs])
        obs_diffs = (obs - centroids_point)
        ss = np.sqrt(((obs - centroids_point)**2)).sum()
        ss_n = ((obs - centroids_point) ** 2).sum()
        sums.append(ss_n)

    assert abs(inertia - np.sum(sums)) < 1, f"Expected {inertia} but got {np.sum(sums)}"

    sums_of_squares_for_differen_n.append(np.sum(sums))

plt.plot(range(2, 10), sums_of_squares_for_differen_n)
plt.show()


