import pandas as pd
import numpy as np
from sklearn import metrics
import sys
import matplotlib.pyplot as plt
reload(sys)
sys.setdefaultencoding('utf8')
from sklearn import preprocessing
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

######################################
##             CONSTANTS            ##
######################################
INPUT_FILE = "movie_metadata.csv" #--do not change
N_PCA = 3 #--do not change
AXIS0_PCA = 0 #--{0, 1, 2}
AXIS1_PCA = 1 #--{0, 1, 2} different from AXIS0_PCA
NUM_MOVIES_PCA = 150 #--keep positive
COLOR = 0 #--0: gross, 1: year, 2: director_facebook_likes, 3: num_critic_for_reviews
PLOT_PCA_2D = True #--{True, False}
PLOT_PCA_3D = True #--{True, False}

######################################
##       FUNCTION DEFINITIONS       ##
######################################
# Read file and return pandas df
def getDFFromFile(f):
	return pd.read_table(f, sep=",")

# Get movie titles
def getTitles(df):
	return df['movie_title']

# Keep only numeric features
def keepNumeric(df):
	return df[[c for c in df if df[c].dtype != np.dtype('O')]]

# Perform imputation -> mean value per feature
def impute(df):
	fill = pd.Series([df[c].mean() for c in df], index=df.columns)
	return df.fillna(fill)

# Transform (scale) data using min-max scaler
def scale(df, scaler):
	return scaler.fit_transform(df)

# Scale feature to [a-b] range
def scale01(col, a=0, b=1):
	return (((col - min(col)) * (b - a)) / (max(col) - min(col))) + a

# perform PCA
def performPCA(n, df):
	pca = PCA(n_components=n)
	pca.fit(df)
	df_tr = pca.transform(df)
	pca_components = pca.components_
	return (pca, df_tr, pca_components)

# create arrays for coloring PCA plots
def makeColors(df):
	colors = []
	colors.append(df['gross'])
	colors.append(df['title_year'])
	colors.append(df['director_facebook_likes'])
	colors.append(df['num_critic_for_reviews'])
	return [scale01(c) for c in colors]

# Plot PCA
def plotPCA(df, df_index, ax1, ax2, num_movies, titles, col):
	max_movies = min(num_movies, len(df)) #--choose the minimum between user input and max allowed
	plt.figure()
	ix = np.array(df_index[df_index.columns[0]].index)
	for i, a in zip(ix[:max_movies], df[:max_movies]):
		r = cm.seismic(col[i])
		plt.scatter(a[ax1], a[ax2], c=r)
		plt.text(a[ax1], a[ax2], titles[i], color=r, fontsize=8)
	plt.xlabel('PCA Axis %d' %ax1)
	plt.ylabel('PCA Axis %d' %ax2)
	plt.show()

# 3D plot for PCA
def plotPCA3D(df, df_index, num_movies, titles, col):
	max_movies = min(num_movies, len(df))#--choose the minimum between user input and max allowed
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ix = np.array(df_index[df_index.columns[0]].index)
	for i, a in zip(ix[:max_movies], df[:max_movies]):
		r = cm.seismic(col[i])
		ax.scatter(a[0], a[1], a[2], c=r)
		ax.text(a[0], a[1], a[2], titles[i], fontsize=8, color=r)
	plt.xlabel('PCA Axis 0')
	plt.ylabel('PCA Axis 1')
	plt.show()

##################################################
##########              MAIN            ##########
##################################################
if __name__ == "__main__":
	movies = getDFFromFile(INPUT_FILE)
	movie_titles = getTitles(movies)
	movies = keepNumeric(movies)
	movies = impute(movies)
	min_max_scaler = preprocessing.MinMaxScaler()
	movies_scaled = scale(movies, min_max_scaler)
	colors = makeColors(movies)

	# perform PCA
	pca, movies_tr, axes = performPCA(N_PCA, movies_scaled)

	# plot PCA
	if PLOT_PCA_2D:
		plotPCA(movies_tr, movies, AXIS0_PCA, AXIS1_PCA, NUM_MOVIES_PCA, movie_titles, colors[COLOR])

	# plot PCA in 3D
	if PLOT_PCA_3D:
		plotPCA3D(movies_tr, movies, NUM_MOVIES_PCA, movie_titles, colors[COLOR])
