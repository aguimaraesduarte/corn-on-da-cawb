import json
import urllib
import untangle


def gimme_json(response):
    data = json.loads(response.read())
    return json.dumps(data, indent=4)


def gimme_xml(response):
    data = response.read()
    data = untangle.parse(data)
    return data


def read_page(URL, opts):
    # urlencode converts the dictionary to a list of x=y pairs
    query_url = URL + urllib.urlencode(opts)
    return urllib.urlopen(query_url)

URL = "http://www.omdbapi.com/?"

json_query = {
    's' : 'bat\w*',
    'r' : 'json',
    'tomatoes' : 'true'
}

xml_query = {
    't' : 'Avatar',
    'r' : 'xml',
    'plot' : 'full'
}

response = read_page(URL, xml_query)
tree = gimme_xml(response)
print tree.children[0].children[0]['director']

