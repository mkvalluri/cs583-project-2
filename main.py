##File: main.py
##Description: Main entry point of the program.
##Author(s): Murali Krishna Valluri, Spoorthi Pendyala
##Date: 06/29/2017

import cs583util as util
import cs583classification as classification

##mode = 0 - train; 1 - test
mode = 0
util.debug_mode = 0

dataFolder = 'data/'
fileNames = [['obama.csv', 'obama_test.csv']]#, ['romney.csv', 'romney_test.csv']]
train_data_raw = []
test_data_raw = []

if mode == 0:
    for fileN in fileNames :
        train_data_raw.append(util.readCSVFile(dataFolder, fileN[0], 1, 4290))
        test_data_raw.append(util.readCSVFile(dataFolder, fileN[0], 5000, 5500))

    for index, data in enumerate(train_data_raw):
        util.printString('File: ' + fileNames[index][0], "File read:")  
        util.printList(data, 'Data read:')  
    
for index, data in enumerate(train_data_raw):
    train_data_raw[index] = util.cleanUpData(data)   

for index, data in enumerate(test_data_raw):
    test_data_raw[index] = util.cleanUpData(data)     

for index, data in enumerate(train_data_raw):
    util.printString('Training Dataset: ' + fileNames[index][0].split('.')[0]) 
    util.printList(data, 'After cleanup:')     

for index, data in enumerate(test_data_raw):
    util.printString('Testing Dataset: ' + fileNames[index][0].split('.')[0]) 
    util.printList(data, 'After cleanup:')  

for tData in train_data_raw:
    labels = tData[0]
    data = tData[1]
    clf = classification.createClassifiers(labels, data)
    
    for ttData in test_data_raw:
        tLabels = ttData[0]
        tData = ttData[1]
        predicted = clf.predict(tData)
        target_names = ['Negative', 'Neutral', 'Positive']
        from sklearn import metrics
        print(metrics.classification_report(tLabels, predicted, target_names=target_names))

        from sklearn.metrics import accuracy_score
        print('Accuracy: ' + "{:.00%}".format(accuracy_score(tLabels, predicted, normalize=True)))