import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial import distance


x = pd.read_csv("Survey/food_deserts/outputs/data.csv")

latitude = x.latitude
longitude = x.longitude
demanda = x.cant_carga * x.num_viajes_lugar_principal

X = list(zip(latitude, longitude))

af = AffinityPropagation(damping=.85).fit(X)
af
metrics.silhouette_score(X, af.labels_)
metrics.calinski_harabasz_score(X, af.labels_)


len(af.cluster_centers_)


plt.scatter(latitude, longitude, c=af.labels_)
