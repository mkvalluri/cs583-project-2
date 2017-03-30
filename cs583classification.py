##File: cs583classification.py
##Description: Classification algorithms
##Author(s): Murali Krishna Valluri, Spoorthi Pendyala
##Date: 06/29/2017

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold

import cs583util as util
            
def createClassifiers(labels, data):
    clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', linear_model.LogisticRegression())])
    clf = clf.fit(data, labels)
    return clf

def printResults(clf, target_names, data):
    tLabels = data[0]
    tData = data[1]
    predicted = clf.predict(tData)
            
    util.printString(metrics.classification_report(tLabels, predicted, target_names=target_names), overrideDebug = 1)
    util.printString('Accuracy: ' + "{:.00%}".format(accuracy_score(tLabels, predicted, normalize=True)), overrideDebug = 1)

def createKCrossFold(labels, data, k = 3):
    skf = StratifiedKFold(n_splits = k)
    sss = StratifiedShuffleSplit(n_splits=k, test_size=0.5, random_state=0)
    kf = KFold(n_splits=k)

    dataSets = []
    
    for train, test in kf.split(data, labels):
        tDataSets = []
        trainingData = []
        trainingLabels = []
        testData = []
        testLabels = []
        
        util.printString("Training: %s \nTesting: %s\n\n" % (train, test))
        
        for index in train:
            trainingData.append(data[index])
            trainingLabels.append(labels[index])

        tDataSets.append([trainingLabels, trainingData])
        
        for index in test:
            testData.append(data[index])
            testLabels.append(labels[index])

        tDataSets.append([testLabels, testData])

        dataSets.append(tDataSets)
    
    return dataSets