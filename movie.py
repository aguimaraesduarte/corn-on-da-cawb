import pandas as pd
import numpy as np
from sklearn import metrics
import sys
import matplotlib.pyplot as plt
reload(sys)
sys.setdefaultencoding('utf8')
from sklearn import preprocessing


movies = pd.read_table("movie_metadata.csv", sep=",")

# reorder columns to have 'gross' as the last column
cols = movies.columns.tolist()
cols = cols[:8]+cols[9:]+['gross']
movies = movies[cols]

# Impute missing data
#fill = pd.Series([movies[c].value_counts().index[0] #most common value
fill = pd.Series(["Missing" #create new label
	if movies[c].dtype == np.dtype('O')
	else movies[c].mean()
	for c in movies], index=movies.columns)
movies = movies.fillna(fill)

# separate into X and Y
X = movies.drop('gross', axis=1)
X_num = movies[[c for c in movies if movies[c].dtype != np.dtype('O')]]
X_str = movies[[c for c in movies if movies[c].dtype == np.dtype('O')]]
Y = movies['gross']

# scale data
X_num_scaled = preprocessing.scale(X_num)

# PCA
from sklearn.decomposition import PCA
n = 3 # number of components we want
pca = PCA(n_components=n)
pca.fit(X_num_scaled)
X2 = pca.transform(X_num_scaled)
pca_components = pca.components_
movies_index = np.array(Y.index)

ax1 = 1
ax2 = 2
max_movies = min(500, len(X2))
plt.figure()
for i, a in zip(movies_index[:max_movies], X2[:max_movies]):
	r = np.random.rand(3)*0.7
	plt.scatter(a[ax1], a[ax2], color=r)
	plt.text(a[ax1], a[ax2], X_str['movie_title'][i], color=r, fontsize=8)

plt.xlim((min([a[ax1] for a in X2[:max_movies]]), max([a[ax1] for a in X2[:max_movies]])))
plt.ylim((min([a[ax2] for a in X2[:max_movies]]), max([a[ax2] for a in X2[:max_movies]])))
plt.xlabel('PCA Axis 0')
plt.ylabel('PCA Axis 2')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, a in zip(movies_index[:max_movies], X2[:max_movies]):
	r = np.random.rand(3)*0.7
	ax.scatter(a[0], a[1], a[2], color=r)
	ax.text(a[0], a[1], a[2], X_str['movie_title'][i], fontsize=8, color=r)

plt.show()