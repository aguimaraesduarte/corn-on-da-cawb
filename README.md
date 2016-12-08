# Readme For Corn-on-da-Cawb ML-1 Project

This file gives order of flow and general instructions for runnging
the scripts and performing the analyses involved in the ML1 final 
project for the group Corn-on-da-Cawb. 

## Genre Prediction


## Revenue Prediction

Simply run `gross_predictions.py` all output is printed to the console,
and some scatter plots will appear.

## Clustering


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



