import pandas as pd
# from matplotlib import pyplot as plt
import numpy as np
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

movies = pd.read_table("movie_metadata.csv", sep=",")
movies['movie_title'] = movies['movie_title'].apply(lambda x: x.replace('\xc2\xa0', ''))

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

movies = pd.get_dummies(movies, columns=['content_rating'])

movies = movies[movies['title_year'] > 2010]

predictors = [
              'cast_total_facebook_likes',
              'title_year',
              'director_facebook_likes',
              'movie_facebook_likes',
              'imdb_score',
              'budget',
              'duration',
              'num_user_for_reviews',
              'num_voted_users',
              'content_rating_Passed',
              'content_rating_R',
              'content_rating_TV-14',
              'content_rating_TV-G',
              'content_rating_TV-MA',
              'content_rating_TV-PG',
              'content_rating_TV-Y',
              'content_rating_TV-Y7',
              'content_rating_Unrated',
              'content_rating_X'
             ]
response = ['gross']


# x_all = movies[predictors]
# y_all = movies[response]
#
# x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=5)

movies_train, movies_test = train_test_split(movies, random_state=5)
x_train = movies_train[predictors]
y_train = movies_train[response]
x_test = movies_test[predictors]
y_test = movies_test[response]

# x_standardizer = StandardScaler()
# x_train = x_standardizer.fit_transform(x_train)
# x_test = x_standardizer.transform(x_test)
#
# y_standardizer = StandardScaler()
# y_train = y_standardizer.fit_transform(y_train)
# y_test = y_standardizer.transform(y_test)


# depths = range(10, 25)
# for depth in depths:
#     model = RandomForestRegressor(max_depth=depth, oob_score=True)
#     model.fit(x_train, y_train)
#     print "Max Depth %s OOB: %.2f" % (depth, model.oob_score_)

models = [
          ('Linear Regression', LinearRegression()),
          ('Random Forest', RandomForestRegressor(max_depth=23)),
          ('Lasso', Lasso()),
          ('Tree', DecisionTreeRegressor(max_depth=5))
         ]

for name, model in models:
    model.fit(x_train, y_train)
    # predictions = model.predict(x_test)
    # print '%s Test MSE: %.2f' % (name, mean_squared_error(predictions, y_test))
    r2 = model.score(x_test, y_test)
    print 'R^2 Test: %.2f' % r2
    #print model.feature_importances_

# MSE TEST IS

