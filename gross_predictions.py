"""
This script runs three iterations of a Random Forest Regressor to predict gross revenue of the movies int he IMDB
movie data.


"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def actor_imdb_score(input_list):
    actor_name = []
    mean_score = []

    for i in input_list:
        data1 = movies.loc[movies['actor_1_name'] == i, 'imdb_score']
        data2 = movies.loc[movies['actor_2_name'] == i, 'imdb_score']
        data3 = movies.loc[movies['actor_3_name'] == i, 'imdb_score']

        data_all = data1.append(data2).append(data3)

        actor_name += [i]
        mean_score += [data_all.mean()]

    return actor_name, mean_score


def add_actor_scores(df):
    actor1 = set(df['actor_1_name'])
    actor2 = set(df['actor_2_name'])
    actor3 = set(df['actor_3_name'])

    actors = np.array(list(actor1.union(actor2).union(actor3)))
    actors = actors[actors != 'nan']

    test_actor_name, test_mean_score = actor_imdb_score(actors)
    actor_df = pd.DataFrame({'actor_name': test_actor_name, 'actor_imdb_score': test_mean_score})

    # -- Merge Actor Data back to original data frame
    df = df.merge(actor_df, left_on='actor_1_name', right_on='actor_name', how='left')
    df = df.merge(actor_df, left_on='actor_2_name', right_on='actor_name', how='left', suffixes=('_1', '_2'))
    df = df.merge(actor_df, left_on='actor_3_name', right_on='actor_name', how='left')
    df.drop(['actor_name_1', 'actor_name_2', 'actor_name'], 1)
    return df


def add_director_scores(df):
    directors = np.array(list(set(df['director_name'])))
    directors = directors[directors != 'nan']

    test_director_name, test_mean_score = director_imdb_score(directors)
    director_df = pd.DataFrame({'director_name': test_director_name, 'director_imdb_score': test_mean_score})
    df = df.merge(director_df, on='director_name', how='left')
    return df


def add_dummies(df):
    df = pd.get_dummies(df, columns=['clean_genre'])
    df = pd.get_dummies(df, columns=['content_rating'])
    return df


def add_features(df):
    df = add_bucketed_genre(df)
    df = add_actor_scores(df)
    df = add_director_scores(df)
    df = add_dummies(df)
    return df


def add_bucketed_genre(df):
    df['clean_genre'] = df['genres'].apply(set_genres)
    return df


def director_imdb_score(input_list):
    director_name = []
    mean_score = []

    for i in input_list:
        data_all = movies.loc[movies['director_name'] == i, 'imdb_score']

        director_name += [i]
        mean_score += [data_all.mean()]

    return director_name, mean_score


def clean_data(df):
    df['movie_title'] = df['movie_title'].apply(lambda x: x.replace('\xc2\xa0', ''))
    df = df[df['type'] == 'movie']
    df = df[df['country'] == 'USA']
    df = df[pd.notnull(df['gross'])]
    return df


def get_best_tree_depth(x_train, y_train, min_depth, max_depth, num_estimators, by=1):
    depths = range(min_depth, max_depth+1, by)
    best_oob = - float('Inf')
    best_depth = min_depth
    for depth in depths:
        model = RandomForestRegressor(max_depth=depth, oob_score=True, n_estimators=num_estimators)
        model.fit(x_train, y_train)
        oob = model.oob_score_
        if oob > best_oob:
            best_oob = oob
            best_depth = depth
    return best_depth, best_oob


def get_top_n_features(forest_model, predictor_list, n):
    feature_scores = forest_model.feature_importances_
    tuple_list = zip(predictor_list, feature_scores)
    tuple_list = sorted(tuple_list, key=lambda tup: tup[1], reverse=True)
    tuple_list = tuple_list[1:n]
    return tuple_list


def set_genres(input_data):
    genreset = set(input_data.split('|'))
    if len(genreset) == 1:
        if len(set(['Documentary', 'Biography', 'Game-Show', 'Reality-TV', 'News']) & genreset) > 0:
            genre = 'True'
        elif len(set(['Crime', 'Mystery', 'Thriller', 'Film-Noir']) & genreset) > 0:
            genre = 'Thriller'
        elif len(set(['Sci-Fi', 'Fantasy', 'Adventure', 'Animation']) & genreset) > 0:
            genre = 'Fantasy'
        elif 'Horror' in genreset:
            genre = 'Horror'
        elif len(set(['Action', 'Western', 'War']) & genreset) > 0:
            genre = 'Action'
        elif len(set(['Drama', 'Romance', 'History']) & genreset) > 0:
            genre = 'Drama'
        elif len(set(['Comedy', 'Music', 'Musical', 'Family']) & genreset) > 0:
            genre = 'Comedy'
        else:
            genre = 'Unknown'
    else:
        if len(set(['Documentary', 'Biography', 'Game-Show', 'Reality-TV', 'News']) & genreset) > 0:
            genre = 'True'
        elif 'Horror' in genreset:
            genre = 'Horror'
        elif len(set(['Sci-Fi', 'Fantasy']) & genreset) > 0:
            genre = 'Fantasy'
        elif len(set(['Western', 'War']) & genreset) > 0:
            genre = 'Action'
        elif 'Thriller' in genreset:
            genre = 'Thriller'
        elif 'Action' in genreset:
            genre = 'Action'
        elif len(set(['Animation', 'Family']) & genreset) > 0:
            genre = 'Comedy'
        elif 'Drama' in genreset and 'Comedy' not in genreset:
            genre = 'Drama'
        elif 'Comedy' in genreset and 'Drama' not in genreset:
            genre = 'Comedy'
        elif len(set(['Drama', 'Romance', 'History']) & genreset) > 0:
            genre = 'Drama'
        else:
            genre = 'Other'
    return genre


def impute(df):
    fill = pd.Series(["Missing"  # create new label
                      if df[c].dtype == np.dtype('O')
                      else df[c].mean()
                      for c in df], index=df.columns)
    df = df.fillna(fill)
    return df


def plot_predicted_observed(observed, predicted):
    correlation = pearsonr(observed, predicted)[0]
    p = plot_scatter_with_text(observed, predicted,
                               title="Actual Gross Revenue vs. Predicted Gross Revenu on Test Set",
                               x_label='Actual Gross Revenue', y_label='Predicted Gross Revenue',
                               text="$R^2$: $%.2f$" % correlation**2)
    return p


def plot_scatter_with_text(x, y, title, x_label, y_label, text):

    x_loc = min(x) * 1.1
    y_loc = max(y) * 0.9

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.text(s=text, x=x_loc, y=y_loc, figure=fig)
    ax1.scatter(x, y, c="b")
    plt.axis('equal')
    axis = plt.axis()
    plt.axis(axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    return plt


def run_models_print_results(model_list, x_train, x_test, y_train, y_test, predictor_list):
    for name, model in model_list:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        print '%s Test MSE: %.2f' % (name, mean_squared_error(predictions, y_test))
        r2 = model.score(x_test, y_test)
        print '%s R^2 Test: %.2f' % (name, r2)
        print 'The top 5 features are: %s' % get_top_n_features(model, predictor_list, 5)
        plot_predicted_observed(y_test, predictions)


def split_test_train(df, predictor_list, response_var, test_size, random_state=None):
    df_train, df_test = train_test_split(df, random_state=random_state, test_size=test_size)
    x_train = df_train[predictor_list]
    y_train = df_train[response_var]
    x_test = df_test[predictor_list]
    y_test = df_test[response_var]
    return x_train, x_test, y_train, y_test


def standardize_df(df):
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)
    return df_std, scaler


def standardize_test_train(x_train, x_test, y_train, y_test):
    x_train, x_scaler = standardize_df(x_train)
    x_test = x_scaler.transform(x_test)

    y_train, y_scaler = standardize_df(y_train)
    y_test = y_scaler.transform(y_test)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':

    movies = pd.read_table("movie_data_plus.csv", sep=",")
    movies = clean_data(movies)
    movies = impute(movies)
    movies = add_features(movies)

    """
    The first model is fit with all predictors and all obervations. This model should give the
    best R^2 of around 0.68 with MSE 1376812281335483.

    """

    predictors = [
        'cast_total_facebook_likes',
        'title_year',
        'director_facebook_likes',
        'movie_facebook_likes',
        'imdb_score',
        'duration',
        'budget',
        'num_user_for_reviews',
        'num_voted_users',
        'content_rating_Missing',
        'content_rating_NC-17',
        'content_rating_Not Rated',
        'content_rating_PG',
        'content_rating_R',
        'content_rating_PG-13',
        'clean_genre_Action',
        'clean_genre_Comedy',
        'clean_genre_Drama',
        'clean_genre_Fantasy',
        'clean_genre_Horror',
        'clean_genre_Thriller',
        'clean_genre_True',
        'metascore',
        'release_month',
        'actor_imdb_score_1',
        'actor_imdb_score_2',
        'actor_imdb_score',
        'director_imdb_score'
    ]
    response = 'gross'

    train_x, test_x, train_y, test_y = split_test_train(movies, predictors, response, random_state=None, test_size=0.3)
    best_depth, best_oob = get_best_tree_depth(train_x, train_y, 10, 30, 15)
    models = [('Random Forest', RandomForestRegressor(max_depth=best_depth))]
    print "\nThe first model is fit with all predictors over all observations: "
    print "The best tree depth is at: %s" % best_depth
    print "Variance of gross over this data set: %s" % np.var(movies['gross'])
    run_models_print_results(models, train_x, test_x, train_y, test_y, predictors)

    """
    The second model takes a more "business sense" approach. We limit the model to only those predictors
    that can be known prior to release.

    """

    predictors = [
        'cast_total_facebook_likes',
        'title_year',
        'director_facebook_likes',
        'duration',
        'budget',
        'content_rating_Missing',
        'content_rating_NC-17',
        'content_rating_Not Rated',
        'content_rating_PG',
        'content_rating_R',
        'content_rating_PG-13',
        'clean_genre_Action',
        'clean_genre_Comedy',
        'clean_genre_Drama',
        'clean_genre_Fantasy',
        'clean_genre_Horror',
        'clean_genre_Thriller',
        'clean_genre_True',
        'release_month',
        'actor_imdb_score_1',
        'actor_imdb_score_2',
        'actor_imdb_score',
        'director_imdb_score'
    ]

    train_x, test_x, train_y, test_y = split_test_train(movies, predictors, response, random_state=None, test_size=0.3)
    best_depth, best_oob = get_best_tree_depth(train_x, train_y, 10, 30, 15)
    models = [('Random Forest', RandomForestRegressor(max_depth=best_depth))]
    print "\nThe second model is fit with only the predictors that are known prior to release. "
    print "The best tree depth is at: %s" % best_depth
    print "Variance of gross over this data set: %s" % np.var(movies['gross'])
    run_models_print_results(models, train_x, test_x, train_y, test_y, predictors)

    """
    The third model is an extension of the second. In this model though, we remove all movies released prior
    to 2010. We remove the outlier movies with gross revenue above $59,000,000 (the rare blockbusters). $59,000,000
    is near the 75th percentile. That is, our data has a left skew.

    """

    movies = movies[movies['title_year'] > 2010]
    movies = movies[movies['gross'] < 59000000]

    predictors = [
        'cast_total_facebook_likes',
        'title_year',
        'director_facebook_likes',
        'duration',
        'budget',
        'content_rating_Missing',
        'content_rating_NC-17',
        'content_rating_Not Rated',
        'content_rating_PG',
        'content_rating_R',
        'content_rating_PG-13',
        'clean_genre_Action',
        'clean_genre_Comedy',
        'clean_genre_Drama',
        'clean_genre_Fantasy',
        'clean_genre_Horror',
        'clean_genre_Thriller',
        'clean_genre_True',
        'release_month',
        'actor_imdb_score_1',
        'actor_imdb_score_2',
        'actor_imdb_score',
        'director_imdb_score'
    ]

    train_x, test_x, train_y, test_y = split_test_train(movies, predictors, response, random_state=None, test_size=0.3)
    best_depth, best_oob = get_best_tree_depth(train_x, train_y, 10, 30, 15)
    models = [('Random Forest', RandomForestRegressor(max_depth=best_depth))]
    print "\nThe third model is fit with only the predictors that are known prior to release on reduced data set. "
    print "The best tree depth is at: %s" % best_depth
    print "Variance of gross over this data set: %s" % np.var(movies['gross'])
    run_models_print_results(models, train_x, test_x, train_y, test_y, predictors)

    """
    It is at this point that we are reminded of some of the fundamentals of modeling. The second model is indeed worse
    than the first. We expect this as we are removing valuable information about the sucess of the movie. Information
    that can only be known after the movie has been released. The R^2 is reduced and the MSE increases. Now, the third
    iteration is a bit more interesting. In the case both R^2 and MSE are reduced. The reason is the total variance
    of the data has been reduced.

    """

    plt.show()




