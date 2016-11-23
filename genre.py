import numpy as np
import csv
from collections import Counter
import random
import time
from sklearn.neighbors import KNeighborsClassifier

start = time.time()

movielist = []

keycounter = Counter()

with open('movie_metadata.csv', 'r') as movies:
    moviereader = csv.reader(movies)
    i = 0
    for line in moviereader:
        i += 1
        title = unicode(line[11], encoding = 'utf-8').replace(u'\xa0', ' ').rstrip()
        genres = line[9]
        keywords = line[16]
        if i > 1:
            genreset = set(genres.split('|'))
            # print genreset
            if len(set(['Documentary', 'Biography','Game-Show','Reality-TV','News']) & genreset) > 0:
                genre = 'True'
            elif len(set(['Crime','Mystery','Thriller','Film-Noir']) & genreset) > 0:
                genre = 'Thriller'
            elif len(set(['Sci-Fi','Fantasy','Adventure','Animation']) & genreset) > 0:
                genre = 'Fantasy'
            elif 'Horror' in genreset:
                genre = 'Horror'
            elif len(set(['Action', 'Western', 'War']) & genreset) > 0:
                genre = 'Action'
            else:
                genre = 'General'
            keylist = keywords.split('|')
            movielist.append([title, genre, keylist])
            for key in keylist:
                keycounter[key] += 1

i = 0

keyfeat = []

for item in keycounter:
    if keycounter[item] > 10:
        i += 1
        # print i
        # print item + ': ' + str(keycounter[item])
        keyfeat.append(item)

# print movielist

random.shuffle(movielist)

# print movielist

def extract(list, keywords):
    titles = []
    X = []
    Y = []
    for item in list:
        titles.append(item[0])
        features = []
        for key in keywords:
            if key in item[2]:
                val = 1
            else:
                val = 0
            features.append(val)
        X.append(features)
        Y.append(item[1])
    X = np.array(X)
    Y = np.array(Y).T
    return titles, X, Y


train = movielist[:-len(movielist)/10]
test = movielist[-len(movielist)/10:]

titles_train, X_train, Y_train = extract(train, keyfeat)
titles_test, X_test, Y_test = extract(test, keyfeat)

print titles_train
print X_train
print Y_train

end = time.time()

print 'Feature extraction: ' + str(end - start) + ' seconds'

def checkpred(obs, pred):
    num = len(obs)
    true = 0
    for i in range(num):
        if obs[i] == pred[i]:
            true += 1
    return float(true)/num

def knn(train_X, train_Y, test_X, test_Y, num):
    bestk = 0
    bestrate = 0.0
    bestpred = []
    for i in range(1,num):
        knnclass = KNeighborsClassifier(n_neighbors = i)
        knnclass.fit(train_X, train_Y)
        pred = knnclass.predict(test_X)
        rate = checkpred(test_Y, pred)
        if rate > bestrate:
            bestk = i
            bestrate = rate
            bestpred = pred
    return (1 - bestrate), bestk, bestpred

knnstart = time.time()

failrate, topk, pred = knn(X_train, Y_train, X_test, Y_test, 19)

print '\n--------------------------\n'

print 'Misclassification rate is ' + str(failrate) + ' for ' + str(topk) + ' neighbors'

knnend = time.time()

print '\n--------------------------\n'

print 'Time for KNN: ' + str(knnend - knnstart) + ' seconds'

print '\n--------------------------\n'

for i in range(len(Y_test)):
    print titles_test[i] + ' - Actual: ' + Y_test[i] + ' | Predicted: ' + pred[i]