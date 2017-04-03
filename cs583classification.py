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
import itertools
            
scores = []

def init():    
    scores.append([]) #obama
    scores.append([]) #romney

    scores[0].append([]) #obama negative
    scores[0][0].extend([[],[],[]]) #obama negative precision, recall, fscore
    scores[0].append([]) #obama positive
    scores[0][1].extend([[],[],[]]) #obama negative precision, recall, fscore    
    scores[0].append([]) #obama accuracy

    scores[1].append([]) #romney negative
    scores[1][0].extend([[],[],[]]) #romney negative precision, recall, fscore
    scores[1].append([]) #romney positive
    scores[1][1].extend([[],[],[]]) #romney negative precision, recall, fscore   
    scores[1].append([]) #romney accuracy

def createClassifiers(labels, data):
    clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', linear_model.LogisticRegression())])
    clf = clf.fit(data, labels)
    return clf

def printResults(clf, target_names, data, idx):
    tLabels = data[0]
    tData = data[1]
    predicted = clf.predict(tData)
    result = metrics.classification_report(tLabels, predicted, target_names=target_names)
    accuracy = accuracy_score(tLabels, predicted, normalize=True)    
	
    util.printString(result)
    util.printString('Accuracy: ' + "{:.00%}".format(accuracy))

    update_metrics(result, idx)
    scores[idx][2].append(accuracy) #accuracy
    
def update_metrics(data, idx1):
	data = data.split('\n')
	td = get_individual_scores(data[2])
	update_individual_scores(idx1, 0, td)
	
	td = get_individual_scores(data[4])
	update_individual_scores(idx1, 1, td)

def get_individual_scores(line):
    line = line.split(' ')
    line = filter(None, line)
    return line

def update_individual_scores(idx1, idx2, data):
    scores[idx1][idx2][0].append(data[1]) #precision
    scores[idx1][idx2][1].append(data[2]) #recall
    scores[idx1][idx2][2].append(data[3]) #f-score

def print_final_metrics():
    print '\n###########################'
    names = ['obama', 'romney']
    class_names = ['Negative', 'Positive']
    
    for idx, score in enumerate(scores):
        print '\n' + names[idx] + ' results:'
        print 'Accuracy: ' + get_average(score[2]) + '\n'
        #loop
        for i in range(0, 2):
            avg_prec = get_average(score[i][0])
            avg_recall = get_average(score[i][1])
            avg_fscore = get_average(score[i][2])
            print class_names[i] + ' Class:'
            print 'Precision: ' + avg_prec
            print 'Recall: ' + avg_recall
            print 'F-Score: ' + avg_fscore + '\n'
			
		
def get_average(data):
    for i, d in enumerate(data):
        data[i] = float(d)
    return "{:.2f}".format((float(sum(data)) / max(len(data), 1)) * 100)

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