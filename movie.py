import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import metrics


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


# PCA
from sklearn.decomposition import PCA
n = 2 # number of components we want
pca = PCA(n_components=n)
pca.fit(X_num)
print pca.components_
print pca.explained_variance_
print pca.explained_variance_ratio_


# KNN Classification
from sklearn.neighbors import KNeighborsClassifier
k = 3 # Can be changed to any integer > 0
algo = KNeighborsClassifier(n_neighbors=k)
algo.fit(X_num, Y)
#hypotheses = algo.predict(test_x)
#print "MSE: %.3f (k = %d)" % (metrics.mean_squared_error(test_y, hypotheses), k)


# Logit
from sklearn.linear_model import LogisticRegression


# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# KNN Regression
from sklearn.neighbors import KNeighborsRegressor


"""

# reshape data into separate columns
color = movies['color'].reshape(-1,1)
director_name = movies['director_name'].reshape(-1,1)
num_critic_for_reviews = movies['num_critic_for_reviews'].reshape(-1,1)
duration = movies['duration'].reshape(-1,1)
director_facebook_likes = movies['director_facebook_likes'].reshape(-1,1)
actor_3_facebook_likes = movies['actor_3_facebook_likes'].reshape(-1,1)
actor_2_name = movies['actor_2_name'].reshape(-1,1)
actor_1_facebook_likes = movies['actor_1_facebook_likes'].reshape(-1,1)
genres = movies['genres'].reshape(-1,1)
actor_1_name = movies['actor_1_name'].reshape(-1,1)
movie_title = movies['movie_title'].reshape(-1,1)
num_voted_users = movies['num_voted_users'].reshape(-1,1)
cast_total_facebook_likes = movies['cast_total_facebook_likes'].reshape(-1,1)
actor_3_name = movies['actor_3_name'].reshape(-1,1)
facenumber_in_poster = movies['facenumber_in_poster'].reshape(-1,1)
plot_keywords = movies['plot_keywords'].reshape(-1,1)
movie_imdb_link = movies['movie_imdb_link'].reshape(-1,1)
num_user_for_reviews = movies['num_user_for_reviews'].reshape(-1,1)
language = movies['language'].reshape(-1,1)
country = movies['country'].reshape(-1,1)
content_rating = movies['content_rating'].reshape(-1,1)
budget = movies['budget'].reshape(-1,1)
title_year = movies['title_year'].reshape(-1,1)
actor_2_facebook_likes = movies['actor_2_facebook_likes'].reshape(-1,1)
imdb_score = movies['imdb_score'].reshape(-1,1)
aspect_ratio = movies['aspect_ratio'].reshape(-1,1)
movie_facebook_likes = movies['movie_facebook_likes'].reshape(-1,1)
gross = movies['gross'].reshape(-1,1)

# perform imputation on missing data
imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
#color = imp.fit_transform(color)
#director_name = imp.fit_transform(director_name)
num_critic_for_reviews = imp.fit_transform(num_critic_for_reviews)
duration = imp.fit_transform(duration)
director_facebook_likes = imp.fit_transform(director_facebook_likes)
actor_3_facebook_likes = imp.fit_transform(actor_3_facebook_likes)
#actor_2_name = imp.fit_transform(actor_2_name)
actor_1_facebook_likes = imp.fit_transform(actor_1_facebook_likes)
#genres = imp.fit_transform(genres)
#actor_1_name = imp.fit_transform(actor_1_name)
#movie_title = imp.fit_transform(movie_title)
num_voted_users = imp.fit_transform(num_voted_users)
cast_total_facebook_likes = imp.fit_transform(cast_total_facebook_likes)
#actor_3_name = imp.fit_transform(actor_3_name)
facenumber_in_poster = imp.fit_transform(facenumber_in_poster)
#plot_keywords = imp.fit_transform(plot_keywords)
#movie_imdb_link = imp.fit_transform(movie_imdb_link)
num_user_for_reviews = imp.fit_transform(num_user_for_reviews)
#language = imp.fit_transform(language)
#country = imp.fit_transform(country)
#content_rating = imp.fit_transform(content_rating)
budget = imp.fit_transform(budget)
title_year = imp.fit_transform(title_year)
actor_2_facebook_likes = imp.fit_transform(actor_2_facebook_likes)
imdb_score = imp.fit_transform(imdb_score)
#aspect_ratio = imp.fit_transform(aspect_ratio)
movie_facebook_likes = imp.fit_transform(movie_facebook_likes)
gross = imp.fit_transform(gross)
"""