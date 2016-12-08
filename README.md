# Readme For Corn-on-da-Cawb ML-1 Project

This file gives order of flow and general instructions for runnging
the scripts and performing the analyses involved in the ML1 final 
project for the group Corn-on-da-Cawb. 

## Genre Prediction
Simply run `genre.py` for genre prediction using plot keywords. Output is printed to the console (Movie title, Actual genre bucket, Predicted genre bucket, List of keywords). Runs LDA only as it's consistently the best model. Other models can be run by uncommenting them in the code.

Simply run `genre_plot.py` for genre prediction using plot description. Output is printed to the console (Movie title, Actual genre bucket, Predicted genre bucket, List of keywords). Runs KNN, Logistic regression, LDA, QDA, Decision tree classification, Random forest classification.

## Revenue Prediction

Simply run `gross_predictions.py` all output is printed to the console,
and some scatter plots will appear.

## Clustering
Simply run `pca.py`. Output is printed as plots:

- scree plot
- cumulative scree plot
- PCA plot (2D)
- PCA plot (3D)

Parameters can be tuned inside the `pca.py` file as such:

- **INPUT_FILE = "movie_metadata.csv"** Input file, should not be changed
- **N_PCA = 3** Number of principal components to keep (should not be higher than 16)
- **AXIS0_PCA = 0** X-axis for PCA plot (2D). One of {0, 1, 2}
- **AXIS1_PCA = 1** Y-axis for PCA plot (2D). One of {0, 1, 2}, different from above
- **NUM_MOVIES_PCA = 150** Number of movies to plot in PCA (clipped to max_movies)
- **COLOR = 1** Coloring of points in PCA plot {0: gross, 1: year, 2: director_facebook_likes, 3: num_critic_for_reviews}
- **PLOT_PCA_2D = True** One of {True, False}
- **PLOT_PCA_3D = True** One of {True, False}

## IMDB Scraping

The scripts necessary to scrape additional data from IMDB 
for information on the movies in the IMDB 5000 data set are `imdb.py` 
and `dict_reader.py`. 
 
The first step is to run `imdb.py` which will output `imdb_elements.txt`
Each line of this file is a string representation of a 
json element. This is an example of one row.

`Element <movie> with attributes {u'plot': u"Ann,...1'} and children []`

The `Element <movie> with attributes` and `and children []` tags can
be removed with a simple find and replace in a text editor. At this 
point you should have a text file where each row is a dictionary. Save
this file as `imdb_movie_dictionaries.txt`. 

Now run the script `imdb.py`. This script reads 
`imdb_movie_dictionaries.txt` and `movie_metadata.csv`. The output of
the script is `imdb_raw_data.csv` which is not used in these experiments
and `movie_data_plus.csv` the resulting of joining `movie_metadata.csv` 
and `imdb_movie_dictionaries.txt`. Apparenetly there are some duplicates
created in the join, but these are removed in the individual model 
scripts. 



