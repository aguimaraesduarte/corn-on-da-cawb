import BeautifulSoup
import json
import random
import time
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

#
# def fetch(url, delay=(2, 5)):
#     """
#     Simulate human random clicking 2.5 seconds then fetch URL.
#     Returns the actual page source fetched and the HTML object.
#     """
#
#     def get_data_from_url(url):
#         # header/value pair is User - Agent: Resistance is futile
#         headers = {'Agent': 'Resistance is futile'}
#         try:
#             req = urllib2.Request(url, headers=headers)
#             data = urllib2.urlopen(req)
#         except ValueError as e:
#             print str(e)
#             return '', BeautifulSoup('', 'html.parser')
#         except:
#             return '', BeautifulSoup('', "html.parser")
#         data = data.read()
#         return data
#
#     time.sleep(random.randint(delay[0], delay[1])) # wait random seconds
#     pagedata = get_data_from_url(url)
#     html = BeautifulSoup(pagedata, "html.parser")
#     return pagedata, html


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
    # time.sleep(random.randint(1, 2))
    return urllib2.urlopen(query_url)


def read_page_from_title(title, url=URL, opts=XML_QUERY):
    opts['t'] = title
    return read_page(url, opts)


