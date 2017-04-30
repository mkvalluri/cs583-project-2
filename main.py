##File: main.py
##Description: Main entry point of the program.
##Author(s): Murali Krishna Valluri, Spoorthi Pendyala
##Date: 06/29/2017

import cs583util as util
import cs583classification as classification

##mode = 0 - train; 1 - test
mode = 1
util.debug_mode = 0

##Declarations, Initializations
dataFolder = 'data/'
fileNames = [['obama.csv', 'obama_test_2.csv'], ['romney.csv', 'romney_test_2.csv']]
train_data_raw = []
test_data_raw = []

##Read training and test data from csv file.
for index, fileN in enumerate(fileNames) :
    train_data_raw.append(util.readFile(dataFolder, fileN[0], 1, 5500))
    if mode == 1:
        test_data_raw.append(util.readFile(dataFolder, fileN[1], 1, 5500))

##Print training data.
for index, data in enumerate(train_data_raw):
    util.printString('File: ' + fileNames[index][0], "File read:")  
    util.printList(data, 'Data read:')  

##Print test data.
for index, data in enumerate(test_data_raw):
    util.printString('File: ' + fileNames[index][1], "File read:")  
    util.printList(data, 'Data read:')  

##Cleanup training data.
for index, data in enumerate(train_data_raw):
    train_data_raw[index] = util.cleanUpData(data)  

##Cleanup test data.
for index, data in enumerate(test_data_raw):
    test_data_raw[index] = util.cleanUpData(data)     

##Print training data after cleanup.
for index, data in enumerate(train_data_raw):
    util.printString('Training Dataset: ' + fileNames[index][0].split('.')[0]) 
    util.printList(data, 'After cleanup:')     

##Print test data after cleanup.
for index, data in enumerate(test_data_raw):
    util.printString('Test Dataset: ' + fileNames[index][1].split('.')[0]) 
    util.printList(data, 'After cleanup:')  

target_names = ['Negative', 'Neutral', 'Positive']
classification.init()

if mode == 0:
    for tIndex, tData in enumerate(train_data_raw):
        print len(tData[0])
        kfoldData = classification.createKCrossFold(tData[0], tData[1], 10)
    
        for kData in kfoldData:
            labels = kData[0][0]
            data = kData[0][1]
            clf = classification.createClassifiers(labels, data)

            util.printString('Results for dataset: ' + fileNames[tIndex][0].split('.')[0])
            classification.printResults(clf, target_names, kData[1], tIndex)
            #for ttData in test_data_raw[tIndex]:
            #classification.printResults(clf, ttData, target_names)
    classification.print_final_metrics()
else:
    #clf_len = classification.initializeClassifiers()

    for tIndex, tData in enumerate(train_data_raw):
        print len(tData[0])

        labels = tData[0]
        data = tData[1]
        
        clf = classification.createClassifiers(labels, data)
        #vct = classification.vectorizeData(labels, data)

        #for i in range(0, clf_len):
            #clf1 = classification.fitData(vct, i)
            #classification.printResults(clf1, target_names, tempTestData, tIndex)

        util.printString('Results for dataset: ' + fileNames[tIndex][0].split('.')[0])
        tempTestData = []
        tempTestData.append(test_data_raw[tIndex][0])
        tempTestData.append(test_data_raw[tIndex][1])
        classification.printResults(clf, target_names, tempTestData, tIndex)
    
    classification.print_final_metrics(writeToFile=True)