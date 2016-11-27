import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
from sklearn.preprocessing import Imputer
from sklearn import metrics

os.chdir('/Users/Brigit/Documents/0_USF/2_Fall_Module/621_ML1/Group Project/corn-on-da-cawb')

movies = pd.read_table("movie_metadata.csv", sep=",")

# reorder columns to have 'gross' as the last column
cols = movies.columns.tolist()
cols = cols[:8]+cols[9:]+['gross']
movies = movies[cols]

movies['log_gross'] = np.log(movies['gross'])
movies['log_budget'] = np.log(movies['budget'])

movies['movie_title'] = movies['movie_title'].apply(lambda x: x.replace('\xc2\xa0',''))


# Impute missing data
#fill = pd.Series([movies[c].value_counts().index[0] #most common value
fill = pd.Series(["Missing" #create new label
	if movies[c].dtype == np.dtype('O')
	else movies[c].mean()
	for c in movies], index=movies.columns)
movies = movies.fillna(fill)
#
# # separate into X and Y
# X = movies.drop('gross', axis=1)
# X_num = movies[[c for c in movies if movies[c].dtype != np.dtype('O')]]
# X_str = movies[[c for c in movies if movies[c].dtype == np.dtype('O')]]
# Y = movies['gross']

#--------------------------------------#
# PLOT DATA							   #
#--------------------------------------#
plot_df = movies.select_dtypes(include=[np.number])

plt.figure()
# plt.scatter(np.log(movies['budget']), np.log(movies['gross']), color = 'black')
plt.scatter(movies['title_year'], np.log(movies['gross']), color = 'black')
plt.show()

plt.figure()
plt.scatter(movies['title_year'], movies['gross'], color = 'black')
plt.show()

plt.figure()
plt.plot(movies['gross'], color = 'black', linestyle='None', marker='o')
# plt.scatter(movies['budget'], movies['gross'], color = 'black')
# plt.scatter(movies['title_year'],movies['gross'])
# plt.hist(movies['title_year'].astype(float).dropna())
plt.show()

print pd.scatter_matrix(plot_df)
print plot_df.columns
# plt.scatter(movies['title_year'],movies['movie_facebook_likes'])
# test = movies['title_year'].astype(float)
# Counter(movies['title_year'])



#--------------------------------------#
# AVERAGE ACTOR SCORE				   #
#--------------------------------------#

actor1 = set(movies['actor_1_name'])
actor2 = set(movies['actor_2_name'])
actor3 = set(movies['actor_3_name'])

actors = np.array(list(actor1.union(actor2).union(actor3)))
actors = actors[actors != 'nan']

def actor_imdb_score(input_list):
	actor_name = []
	mean_score = []

	for i in input_list:
		data1 = movies.loc[movies['actor_1_name'] == i,'imdb_score']
		data2 = movies.loc[movies['actor_2_name'] == i,'imdb_score']
		data3 = movies.loc[movies['actor_3_name'] == i,'imdb_score']

		data_all = data1.append(data2).append(data3)

		actor_name += [i]
		mean_score += [data_all.mean()]

	return actor_name, mean_score

test_actor_name, test_mean_score  = actor_imdb_score(actors)
actor_df = pd.DataFrame({'actor_name':test_actor_name, 'actor_imdb_score' : test_mean_score})

# -- Merge Actor Data back to original data frame
movies = movies.merge(actor_df, left_on='actor_1_name', right_on='actor_name', how='left')
movies = movies.merge(actor_df, left_on='actor_2_name', right_on='actor_name', how='left', suffixes=('_1','_2'))
movies = movies.merge(actor_df, left_on='actor_3_name', right_on='actor_name', how='left')
movies.drop(['actor_name_1','actor_name_2', 'actor_name'],1)

#--------------------------------------#
# AVERAGE DIRECTOR SCORE			   #
#--------------------------------------#

directors = np.array(list(set(movies['director_name'])))
directors = directors[directors != 'nan']

def director_imdb_score(input_list):
	director_name = []
	mean_score = []

	for i in input_list:
		data_all = movies.loc[movies['director_name'] == i,'imdb_score']

		director_name += [i]
		mean_score += [data_all.mean()]

	return director_name, mean_score

test_director_name, test_mean_score  = director_imdb_score(directors)
director_df = pd.DataFrame({'director_name':test_director_name, 'director_imdb_score' : test_mean_score})
movies = movies.merge(director_df, on='director_name', how='left')


# #------------CHECKS-------------------------#
# z = [i for i in range(len(test_director_name)) if test_director_name[i] == 'James Cameron']
# test_mean_score[z[0]]
# x = [t for t in test_actor_name if t == 'Orlando Bloom']
# z = [i for i in range(len(test_actor_name)) if test_actor_name[i] == 'Orlando Bloom']
# test_mean_score[z[0]]


#--------------------------------------#
# 0/1 ENCODING ACTOR NAMES   	       #
#--------------------------------------#


all_actors = []
for i in range(len(movies['actor_1_name'])):
	all_actors += [[movies['actor_1_name'][i], movies['actor_2_name'][i], movies['actor_3_name'][i]]]

all_actors = pd.Series(all_actors)

for a in actors:
	z = all_actors.apply( lambda x: int(a in x))
	movies[a] = z

# -------- CHECK -------------------------------------#
# movies[movies['movie_title'] == 'Spectre'].nonzero()
# df.ix[:, (df == 0).all()]
#
# spectre = movies[movies['movie_title'] == 'Spectre']
# for i in range(len(spectre.columns)):
# 	if (spectre.iloc[:,i] !=0).all():
# 		print spectre.iloc[:,i]
# -------- CHECK -------------------------------------#

#--------------------------------------#
# 0/1 ENCODING ACTOR NAMES   	       #
#--------------------------------------#

train, test = train_test_split(movies, test_size=0.3, random_state=0)
train.shape
test.shape

train_Y = train['gross']
train_X = train.drop(['gross','log_gross'], axis=1).select_dtypes(include=[np.number])

test_Y = test['gross']
test_X = test.drop(['gross','log_gross'], axis=1).select_dtypes(include=[np.number])


def model_good(pred_Y_train, pred_Y_test):
	# Test MSE
	test_mse = mean_squared_error(test_Y, pred_Y_test)
	train_mse = mean_squared_error(train_Y, pred_Y_train)

	# Rsquared
	test_r2 = pearsonr(test_Y, pred_Y_test)[0] ** 2
	train_r2 = pearsonr(train_Y, pred_Y_train)[0] ** 2

	print "Test MSE:", test_mse, "Train MSE:", train_mse
	print test_mse / train_mse
	print "Test R2:", test_r2, "Train R2:", train_r2

def make_a_tree(model):
	dot_data = tree.export_graphviz(model, out_file=None,
									feature_names=train_X.columns.values,
									filled=True, rounded=True,
									special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data)
	graph.write_pdf('tree.pdf')

# -----------------------------------------------------------------#
# DECISION TREE													   #
# -----------------------------------------------------------------#



depth = 5 # Note that depth, not alpha, is the hyperparameter
regressor = DecisionTreeRegressor(max_depth=depth)
regressor.fit(train_X, train_Y)
pred_train_Y = regressor.predict(train_X)
predicted_Y = regressor.predict(test_X)



#Look At Tree

model_good(pred_train_Y, predicted_Y)
make_a_tree(regressor)

def clean_feat_importance(model):
	feat_importance = pd.Series(model.feature_importances_)
	feat_name = pd.Series(train_X.columns.values)
	feat_import = pd.DataFrame({'feature': feat_name, 'import':feat_importance})
	print feat_import.sort('import')
# -----------------------------------------------------------------#
# RANDOM FOREST												   #
# -----------------------------------------------------------------#


estimators = 10 # Number of estimators, defaults to 10
# for i in range(1,10):
forest = RandomForestRegressor(n_estimators=estimators, max_depth=4, oob_score=True)
forest.fit(train_X, train_Y)

pred_train_Y = forest.predict(train_X)
predicted_Y = forest.predict(test_X)

print "Max Depth =", i
print "Forest OOB score", 	forest.oob_score_
print clean_feat_importance(forest)
# model_good(pred_train_Y, predicted_Y)

make_a_tree(forest)

forest.estimators_ #List of Trees


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