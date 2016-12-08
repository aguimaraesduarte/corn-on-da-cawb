"""
This script scrapes all available movie data from IMDB 5000 and writes to a txt file. Each movie will be
written to the file as an Element. The Element tag can then be removed for further processing if desired.

"""

import json
import pandas as pd
import urllib
import urllib2
import untangle


JSON_QUERY = {
    's': 'bat\w*',
    'r': 'json',
    'tomatoes': 'true'
}

XML_QUERY = {
    't': '?',
    'r': 'xml',
    'plot': 'full'
}

URL = "http://www.omdbapi.com/?"


def get_single_factor(title, factor):
    """
    Returns the information on the factor for the single movie title
    :param title: a movie title
    :param factor: a single factor of interest
    :return: data on the factor for the movie
    """
    response = read_page_from_title(title)
    tree = gimme_xml(response)
    return tree.children[0].children[0][factor]


def get_series_from_factor(titles, factor):
    """
    Builds a Pandas series of returned information for the single for all movies in titles
    :param titles: a list of movie titles
    :param factor: a single bit of information to pull
    :return:
    """
    factor_series = titles.apply(lambda title: get_single_factor(title, factor))
    return factor_series


def gimme_json(response):
    """
    Given a urllib2 object returns a json tree.
    :param response: urllib2 object
    :return: untangle tree represenation of json
    """
    data = json.loads(response.read())
    return json.dumps(data, indent=4)


def gimme_xml(response):
    """
    Given a urllib2 object returns a untangle XML tree.
    :param response: urllib2 object
    :return: untangle tree represenation of xml
    """
    data = response.read()
    data = untangle.parse(data)
    return data


def read_page(url, opts):
    """
    Request data and returns urllib2 object using given url template and options.

    Note, urlencode converts the dictionary to a list of x=y pairs
    :param url: url to request
    :param opts: the options to append to the request
    :return: urllib2 object
    """

    query_url = url + urllib.urlencode(opts)
    return urllib2.urlopen(query_url)


def read_page_from_title(title, url=URL, opts=XML_QUERY):
    """
    Request all information from IMDB on given title.
    :param title: movie title to find
    :param url: the url template string
    :param opts: the url options
    :return: urllib2 object representation of the page
    """
    opts['t'] = title
    return read_page(url, opts)


if __name__ == '__main__':

    movies = pd.read_table("movie_metadata.csv", sep=",")
    movies['movie_title'] = movies['movie_title'].apply(lambda x: x.replace('\xc2\xa0', ''))

    # we only want information on the IMDB 5000 movies, so we get a list of those titles to retrieve.
    titles = list(movies['movie_title'])
    for i, title in enumerate(titles):
        imdb_response = read_page_from_title(title)
        tree = gimme_xml(imdb_response)
        fd = open('imdb_elements.txt', 'a')
        data = str(tree.children[0].children[0]) + '\n'
        # uncomment this line to write to the file
        # fd.write(data)

