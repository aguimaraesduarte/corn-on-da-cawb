import numpy as np
import csv
from collections import Counter
import random
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

start = time.time()

movielist = []

keycounter = Counter()

genrecounter = Counter()
rawgenrecounter = Counter()

with open('movie_metadata.csv', 'r') as movies:
    moviereader = csv.reader(movies)
    i = 0
    for line in moviereader:
        i += 1
        title = unicode(line[11], encoding = 'utf-8').replace(u'\xa0', ' ').rstrip()
        genres = line[9]
        rawgenrecounter[genres] += 1
        keywords = line[16]
        if i > 1:
            genreset = set(genres.split('|'))
            if len(genreset) == 1:
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
                elif len(set(['Drama', 'Romance', 'History']) & genreset) > 0:
                    genre = 'Drama'
                elif len(set(['Comedy', 'Music', 'Musical', 'Family']) & genreset) > 0:
                    genre = 'Comedy'
                else:
                    genre = 'Unknown'
            else:
                if len(set(['Documentary', 'Biography','Game-Show','Reality-TV','News']) & genreset) > 0:
                    genre = 'True'
                elif 'Horror' in genreset:
                    genre = 'Horror'
                elif len(set(['Sci-Fi','Fantasy']) & genreset) > 0:
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
                elif len(set(['Drama','Romance', 'History']) & genreset) > 0:
                    genre = 'Drama'
                else:
                    genre = 'Other'
            genrecounter[genre] += 1
            keylist = keywords.split('|')
            movielist.append([title, genre, keylist, list(genreset)])
            for key in keylist:
                keycounter[key] += 1

print genrecounter
#print rawgenrecounter

i = 0

keyfeat = []

for item in keycounter:
    if keycounter[item] > 5:
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

def logistic(train_X, train_Y, test_X, test_Y):
    logreg = LogisticRegression()
    logreg.fit(train_X, train_Y)
    pred = logreg.predict(test_X)
    rate = checkpred(test_Y, pred)
    return (1 - rate), pred

def lindisc(train_X, train_Y, test_X, test_Y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_X, train_Y)
    pred = lda.predict(test_X)
    rate = checkpred(test_Y, pred)
    return (1 - rate), pred

def quadpred(train_X, train_Y, test_X, test_Y):
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(train_X, train_Y)
    pred = qda.predict(test_X)
    rate = checkpred(test_Y, pred)
    return (1 - rate), pred

'''
failrates = []

knnstart = time.time()

failrate, topk, knnpred = knn(X_train, Y_train, X_test, Y_test, 19)

failrates.append(failrate)

print '\n--------------------------\n'

print 'Misclassification rate: ' + str(failrate) + ' for ' + str(topk) + ' neighbors\n'

knnend = time.time()

print 'Time for KNN: ' + str(knnend - knnstart) + ' seconds'
'''
print '\n--------------------------\n'

logstart = time.time()

failrate, logpred = logistic(X_train, Y_train, X_test, Y_test)

# failrates.append(failrate)

logend = time.time()

print 'Misclassification rate: ' + str(failrate) + '\n'

print 'Time for Logistic Regression: ' + str(logend - logstart) + ' seconds'

print '\n--------------------------\n'
'''
LDAstart = time.time()

failrate, LDApred = lindisc(X_train, Y_train, X_test, Y_test)

failrates.append(failrate)

LDAend = time.time()

print 'Misclassification rate: ' + str(failrate) + '\n'

print 'Time for Linear Discriminant Analysis: ' +str(LDAend - LDAstart) + ' seconds'

print '\n--------------------------\n'

QDAstart = time.time()

failrate, QDApred = quadpred(X_train, Y_train, X_test, Y_test)

QDAend = time.time()

print 'Misclassification rate: ' + str(failrate) + '\n'

print 'Time for Quadratic Discriminant Analysis: ' + str(QDAend - QDAstart) + ' seconds'

print '\n--------------------------\n'

best = np.argmin(failrates)

if best == 0:
    bestpred = knnpred
    print 'KNN wins'
elif best == 1:
    bestpred = logpred
    print 'Logistic Regression wins'
elif best == 2:
    bestpred = LDApred
    print 'LDA wins'
elif best == 3:
    bestpred = QDApred
    print 'QDA wins'
else:
    print 'Yikes'
'''

metavalid = 0

bestpred = logpred

for i in range(len(Y_test)):
    print titles_test[i] + ' - Actual: ' + Y_test[i] + ' | Predicted: ' + bestpred[i]
    # print test[i]
    if bestpred[i] != Y_test[i] and bestpred[i] in test[i][3]:
        print 'Prediction in description'
        metavalid += 1

print 'Exact match rate: ' + str(1- failrate)
print 'Obscured prediction rate: ' + str(float(metavalid)/len(Y_test))           # predictions that did not match the bucket but did match a
                                                                                 # genre in the original description