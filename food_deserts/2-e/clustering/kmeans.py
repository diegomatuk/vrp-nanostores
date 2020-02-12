import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial import distance


x = pd.read_csv("Survey/vrp-nanostores/vrp-nanostores/food_deserts/outputs/data.csv")

latitude = x.latitude
longitude = x.longitude
demanda = x.cant_carga

X = list(zip(latitude, longitude))


def scoreSilouette(X):
    scores = []
    for i in range(2, 11):
        fitx = KMeans(n_clusters=i, init='k-means++', n_init=5).fit(X)
        score = metrics.silhouette_score(X, fitx.labels_)
        scores.append(score)
    return np.array(scores)


# OPTIMAL NUMBER OF CLUSTERS --> 8
scoreSilouette(X)


def calinski_harabazs(X):
    scores = []
    for i in range(2, 11):
        fitx = KMeans(n_clusters=i, init='k-means++', n_init=5).fit(X)
        score = metrics.calinski_harabasz_score(X, fitx.labels_)
        scores.append(score)
    return np.array(scores)


calinski_harabazs(X)


plt.plot([i for i in range(2, 11)], scoreSilouette(X))
plt.plot([i for i in range(2, 11)], calinski_harabazs(X))

# KMeans ALgorithm wins :)
km = KMeans(n_clusters=8).fit(X)
metrics.silhouette_score(X, km.labels_)
km.labels_
# GROUPING THE DATA TOGHETER
df = pd.DataFrame(list(zip(latitude, longitude, demanda)))
df.columns = ["latitude", "longitude", "demanda"]
df["cluster"] = km.labels_
colores = sns.color_palette()[0:5]
df = df.sort_values("cluster")


list_keys = [str(i) for i in range(0, 8)]
list_values = ["clust_" + str(i) for i in range(1, 9)]

nombres = dict(zip(list_keys, list_values))
nombres
# RENAMING TH VALUES
df["cluster_name"] = [nombres[str(i)] for i in df.cluster]
df


def clusters():
    c1 = df[df.cluster == 0]
    c2 = df[df.cluster == 1]
    c3 = df[df.cluster == 2]
    c4 = df[df.cluster == 3]
    c5 = df[df.cluster == 4]
    c6 = df[df.cluster == 5]
    c7 = df[df.cluster == 6]
    c8 = df[df.cluster == 7]
    k_coords = km.cluster_centers_
    return c1, c2, c3, c4, c5, c6, c7, c8, k_coords


c1, c2, c3, c4, c5, c6, c7, c8, center = clusters()

demand0 = df.loc[df["cluster"] == 0, "demanda"].sum()
demand1 = df.loc[df["cluster"] == 1, "demanda"].sum()
demand2 = df.loc[df["cluster"] == 2, "demanda"].sum()
demand3 = df.loc[df["cluster"] == 3, "demanda"].sum()
demand4 = df.loc[df["cluster"] == 4, "demanda"].sum()
demand5 = df.loc[df["cluster"] == 5, "demanda"].sum()
demand6 = df.loc[df["cluster"] == 6, "demanda"].sum()
demand7 = df.loc[df["cluster"] == 7, "demanda"].sum()

demands = np.array([demand0, demand1, demand2, demand3, demand4, demand5, demand6, demand7])

# WHAT ARE THE BEST MARKETS?


def clusterisar(cluster, indice):
    X = [(center[indice, :])]  # "The center of the index cluster"
    coords = []
    for index, row in cluster.iterrows():
        coords.append((row[0], row[1]))
    # Which  market is nearest to de  center xdxd
    distancia = distance.cdist(X, coords, "euclidean")
    return distancia


c = [c1, c2, c3, c4, c5, c6, c7, c8]
markets = []
for ix, i in enumerate(c):
    a = i.iloc[np.argmin(np.min(clusterisar(i, ix), axis=0))][[
        "latitude", "longitude", "cluster", "cluster_name"]]
    markets.append(a)
markets = np.array(markets)

markets = np.concatenate((markets, demands[:, None]), axis=1)


for cluster in c:
    print(cluster.shape)

for cluster in c:
    print(max(cluster["demanda"]))


markets

for ix, cluster in enumerate(c):
    cluster = np.vstack((markets[ix], cluster))
    # print(cluster)


cluster1 = np.vstack((markets[0], c1))
cluster2 = np.vstack((markets[1], c2))
cluster3 = np.vstack((markets[2], c3))
cluster4 = np.vstack((markets[3], c4))
cluster5 = np.vstack((markets[4], c5))
cluster6 = np.vstack((markets[5], c6))
cluster7 = np.vstack((markets[6], c7))
cluster8 = np.vstack((markets[7], c8))


clusters = [cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8]

clusters_new = []
for i in clusters:
    i = pd.DataFrame(i)
    i.columns = ["latitude", "longitude", 'demanda', 'cluster', 'cluster_name']
    clusters_new.append(i)


clusters_new[0].to_csv(
    "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster1.csv")
clusters_new[1].to_csv(
    "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster2.csv")
clusters_new[2].to_csv(
    "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster3.csv")
clusters_new[3].to_csv(
    "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster4.csv")
clusters_new[4].to_csv(
    "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster5.csv")
clusters_new[5].to_csv(
    "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster6.csv")
clusters_new[6].to_csv(
    "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster7.csv")
clusters_new[7].to_csv(
    "Survey/vrp-nanostores/vrp-nanostores/food_deserts/2-e/clustering/cluster8.csv")
