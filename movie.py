import pandas as pd
import numpy as np
from sklearn import metrics
import sys
import matplotlib.pyplot as plt
reload(sys)
sys.setdefaultencoding('utf8')
from sklearn import preprocessing
import matplotlib.cm as cm


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
X_num = X[[c for c in X if X[c].dtype != np.dtype('O')]]
X_str = X[[c for c in X if X[c].dtype == np.dtype('O')]]
Y = movies['gross']

# scale data
#X_num_scaled = preprocessing.scale(X_num) #throws weird warning
min_max_scaler = preprocessing.MinMaxScaler()
X_num_scaled = min_max_scaler.fit_transform(X_num)
Y_scaled = (((Y - min(Y)) * (1 - 0)) / (max(Y) - min(Y))) + 0

# test
#X_num2 = X_num[["duration","budget","imdb_score"]]
X_num2 = X_num[["num_critic_for_reviews","director_facebook_likes","actor_1_facebook_likes",
                "actor_2_facebook_likes","actor_3_facebook_likes","actor_1_facebook_likes",
                "num_voted_users","num_user_for_reviews","movie_facebook_likes"]]
min_max_scaler = preprocessing.MinMaxScaler()
X_num_scaled2 = min_max_scaler.fit_transform(X_num2)

# PCA
from sklearn.decomposition import PCA
n = 3 # number of components we want
pca = PCA(n_components=n)
pca.fit(X_num_scaled)
X2 = pca.transform(X_num_scaled)
pca_components = pca.components_
movies_index = np.array(Y.index)

ax1 = 0
ax2 = 1
max_movies = min(2500, len(X2))
plt.figure()
for i, a in zip(movies_index[:max_movies], X2[:max_movies]):
	r = cm.seismic(Y_scaled[i])
	plt.scatter(a[ax1], a[ax2], color=r)
	plt.text(a[ax1], a[ax2], X_str['movie_title'][i], color=r, fontsize=8)

plt.xlim((min([a[ax1] for a in X2[:max_movies]]), max([a[ax1] for a in X2[:max_movies]])))
plt.ylim((min([a[ax2] for a in X2[:max_movies]]), max([a[ax2] for a in X2[:max_movies]])))
plt.xlabel('PCA Axis %d' %ax1)
plt.ylabel('PCA Axis %d' %ax2)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
max_movies = min(500, len(X2))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, a in zip(movies_index[:max_movies], X2[:max_movies]):
	r = cm.seismic(Y_scaled[i])
	ax.scatter(a[0], a[1], a[2], color=r)
	ax.text(a[0], a[1], a[2], X_str['movie_title'][i], fontsize=8, color=r)

plt.show()