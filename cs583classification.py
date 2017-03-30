##File: cs583classification.py
##Description: Classification algorithms
##Author(s): Murali Krishna Valluri, Spoorthi Pendyala
##Date: 06/29/2017

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

def createClassifiers(labels, data):
    clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', DecisionTreeClassifier())])
    clf = clf.fit(data, labels)
    return clf


