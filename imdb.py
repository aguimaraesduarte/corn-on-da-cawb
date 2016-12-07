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
    response = read_page_from_title(title)
    tree = gimme_xml(response)
    return tree.children[0].children[0][factor]


def get_series_from_factor(titles, factor):
    factor_series = titles.apply(lambda title: get_single_factor(title, factor))
    return factor_series


def gimme_json(response):
    data = json.loads(response.read())
    return json.dumps(data, indent=4)


def gimme_xml(response):
    data = response.read()
    data = untangle.parse(data)
    return data


def read_page(url, opts):
    # urlencode converts the dictionary to a list of x=y pairs
    query_url = url + urllib.urlencode(opts)
    return urllib2.urlopen(query_url)


def read_page_from_title(title, url=URL, opts=XML_QUERY):
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

