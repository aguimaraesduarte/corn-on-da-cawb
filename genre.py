# MSAN 621 group project
# Corn on da CAWB
# Movie genre prediction by keywords
# Claire Broad + Andre Duarte


import numpy as np
import csv
from collections import Counter
import random
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def extract(list, keywords):
    '''
    :param list: nested lists -- each list is a movie -- [title, genre bucket, [list of five keywords]]
    :param keywords: list of top keywords (features)
    :return: list of titles (titles), array where rows are movies and columns are keywords (X), list/array of genres by bucket (Y)
    '''
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


def checkpred(obs, pred):
    '''
    :param obs: observed genre bucket
    :param pred: predicted genre bucket
    :return: accuracy rate of genre predictions
    '''
    num = len(obs)
    true = 0
    for i in range(num):
        if obs[i] == pred[i]:
            true += 1
    return float(true)/num
"""
def knn(train_X, train_Y, test_X, test_Y, num):
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :param num: k range
    :return: misclassification rate for best iterations, k value for best iteration, prediction list for best iteration
    '''
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
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :return: misclassification rate, prediction list
    '''
    logreg = LogisticRegression()
    logreg.fit(train_X, train_Y)
    pred = logreg.predict(test_X)
    rate = checkpred(test_Y, pred)
    return (1 - rate), pred

def lindisc(train_X, train_Y, test_X, test_Y):
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :return: misclassification rate, prediction list
    '''
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_X, train_Y)
    pred = lda.predict(test_X)
    rate = checkpred(test_Y, pred)
    return (1 - rate), pred

def quadpred(train_X, train_Y, test_X, test_Y):
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :return: misclassification rate, prediction list
    '''
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(train_X, train_Y)
    pred = qda.predict(test_X)
    rate = checkpred(test_Y, pred)
    return (1 - rate), pred

def treepred(train_X, train_Y, test_X, test_Y):
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :return: misclassification rate, prediction list
    '''
    tree = DecisionTreeClassifier(max_depth=30)
    tree.fit(train_X, train_Y)
    pred = tree.predict(test_X)
    rate = checkpred(test_Y, pred)
    return (1 - rate), pred
"""

def knn(train_X, train_Y, num):
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :param num: k range
    :return: misclassification rate for best iterations, k value for best iteration, prediction list for best iteration
    '''
    bestk = 0
    bestrate = 0.0
    bestpred = []
    for i in range(1,num):
        knnclass = KNeighborsClassifier(n_neighbors = i)
        knnclass.fit(train_X, train_Y)
        rate = (1-cross_val_score(knnclass, train_X, train_Y, cv=5,
                       scoring=metrics.make_scorer(metrics.accuracy_score)).mean())
        if rate > bestrate:
            bestk = i
            bestrate = rate
    knnclass = KNeighborsClassifier(n_neighbors = bestk)
    knnclass.fit(train_X, train_Y)
    cv = cross_val_score(knnclass, train_X, train_Y, cv=10,
                       scoring=metrics.make_scorer(metrics.accuracy_score))
    return (1 - cv.mean()), bestk

def logistic(train_X, train_Y):
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :return: misclassification rate, prediction list
    '''
    logreg = LogisticRegression()
    logreg.fit(train_X, train_Y)
    cv = cross_val_score(logreg, train_X, train_Y, cv=10,
                       scoring=metrics.make_scorer(metrics.accuracy_score))
    return (1 - cv.mean())

def lindisc(train_X, train_Y):
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :return: misclassification rate, prediction list
    '''
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_X, train_Y)
    cv = cross_val_score(lda, train_X, train_Y, cv=10,
                       scoring=metrics.make_scorer(metrics.accuracy_score))
    return (1 - cv.mean())

def quadpred(train_X, train_Y):
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :return: misclassification rate, prediction list
    '''
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(train_X, train_Y)
    cv = cross_val_score(qda, train_X, train_Y, cv=10,
                       scoring=metrics.make_scorer(metrics.accuracy_score))
    return (1 - cv.mean())

def treepred(train_X, train_Y):
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :return: misclassification rate, prediction list
    '''
    tree = DecisionTreeClassifier(max_depth=25)
    tree.fit(train_X, train_Y)
    cv = cross_val_score(tree, train_X, train_Y, cv=10,
                       scoring=metrics.make_scorer(metrics.accuracy_score))
    return (1 - cv.mean())

def forestpred(train_X, train_Y):
    '''
    :param train_X: training set feature array
    :param train_Y: training set genre list
    :param test_X: test set feature array
    :param test_Y: test set genre list
    :return: misclassification rate, prediction list
    '''
    forest = RandomForestClassifier(n_estimators=25)
    forest.fit(train_X, train_Y)
    cv = cross_val_score(forest, train_X, train_Y, cv=10,
                       scoring=metrics.make_scorer(metrics.accuracy_score))
    return (1 - cv.mean())



start = time.time()

movielist = []

keycounter = Counter()                                # counter for all keywords in raw data

genrecounter = Counter()                              # counter for genre buckets
rawgenrecounter = Counter()                           # counter for all raw genre tags

titlelist = []

with open('movie_metadata.csv', 'r') as movies:
    moviereader = csv.reader(movies)
    i = 0
    for line in moviereader:
        i += 1
        title = unicode(line[11], encoding = 'utf-8').replace(u'\xa0', ' ').rstrip()
        if title in titlelist:
            pass
        else:
            genres = line[9]
            rawgenrecounter[genres] += 1
            keywords = line[16]
            if i > 1:
                '''
                Bucket movie genres based on the list of genres given in the raw data
                '''
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

# print genrecounter
# print rawgenrecounter

i = 0

keyfeat = []                           # list of keywords with highest prevalence

for item in keycounter:
    if keycounter[item] > 10:
        i += 1
        # print i
        # print item + ': ' + str(keycounter[item])
        keyfeat.append(item)

# print movielist

random.shuffle(movielist)                        # randomize the list for train/test subsetting

# print movielist

titles_train, X_train, Y_train = extract(movielist, keyfeat)

# print titles_train
# print X_train
# print Y_train
#print Y_test

end = time.time()

print 'Feature extraction: ' + str(end - start) + ' seconds'
'''

failrates = []

knnstart = time.time()

failrate, topk = knn(X_train, Y_train, 19)

failrates.append(failrate)

print '\n--------------------------\n'

print 'Misclassification rate: ' + str(failrate) + ' for ' + str(topk) + ' neighbors\n'

knnend = time.time()

print 'Time for KNN: ' + str(knnend - knnstart) + ' seconds'

print '\n--------------------------\n'

logstart = time.time()

failrate = logistic(X_train, Y_train)

failrates.append(failrate)

logend = time.time()

print 'Misclassification rate: ' + str(failrate) + '\n'

print 'Time for Logistic Regression: ' + str(logend - logstart) + ' seconds'

print '\n--------------------------\n'

LDAstart = time.time()

failrate = lindisc(X_train, Y_train)

failrates.append(failrate)

LDAend = time.time()

print 'Misclassification rate: ' + str(failrate) + '\n'

print 'Time for Linear Discriminant Analysis: ' +str(LDAend - LDAstart) + ' seconds'

print '\n--------------------------\n'

QDAstart = time.time()

failrate = quadpred(X_train, Y_train)

failrates.append(failrate)

QDAend = time.time()

print 'Misclassification rate: ' + str(failrate) + '\n'

print 'Time for Quadratic Discriminant Analysis: ' + str(QDAend - QDAstart) + ' seconds'

print '\n--------------------------\n'

treestart = time.time()

failrate = treepred(X_train, Y_train)

failrates.append(failrate)

treeend = time.time()

print 'Misclassification rate: ' + str(failrate) + '\n'

print 'Time for Decision Tree Classifier: ' + str(treeend - treestart) + ' seconds'

print '\n--------------------------\n'

foreststart = time.time()

failrate = forestpred(X_train, Y_train)

failrates.append(failrate)

forestend = time.time()

print 'Misclassification rate: ' + str(failrate) + '\n'

print 'Time for Random Forest Classifier: ' + str(forestend - foreststart) + ' seconds'

print '\n--------------------------\n'

best = np.argmin(failrates)

if best == 0:
    print 'KNN wins'
elif best == 1:
    print 'Logistic Regression wins'
elif best == 2:
    print 'LDA wins'
elif best == 3:
    print 'QDA wins'
elif best == 4:
    print "Decision tree wins"
elif best == 5:
    print "Forest wins"
else:
    print 'Yikes'

metavalid = 0
'''

train = movielist[:-len(movielist)/10]           # first 9/10 of the shuffled list for training
test = movielist[-len(movielist)/10:]            # last 1/10 for testing

titles_train, X_train, Y_train = extract(train, keyfeat)
titles_test, X_test, Y_test = extract(test, keyfeat)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
logpred = logreg.predict(X_test)

metavalid = 0
acc = 0

for i in range(len(Y_test)):
    print titles_test[i] + ' - Actual: ' + Y_test[i] + ' | Predicted: ' + logpred[i]
    if Y_test[i] == logpred[i]:
        acc += 1
    print 'Keywords:'
    for item in test[i][2]:
        print '\t' + item
    if logpred[i] != Y_test[i] and logpred[i] in test[i][3]:
        print '\t\tPrediction in description'
        metavalid += 1
    print '\n-----------------------------------\n'

print 'Exact match rate: ' + str(float(acc)/len(Y_test))
print 'Obscured prediction rate: ' + str(float(metavalid)/len(Y_test))           # predictions that did not match the bucket but did match a
                                                                                 # genre in the original description
