##File: main.py
##Description: Main entry point of the program.
##Author(s): Murali Krishna Valluri, Spoorthi Pendyala
##Date: 06/29/2017

import cs583util as util
import cs583classification as classification

##mode = 0 - train; 1 - test
mode = 0
util.debug_mode = 0

##Declarations, Initializations
dataFolder = 'data/'
fileNames = [['obama.csv', 'obama_test.csv'], ['romney.csv', 'romney_test.csv']]
train_data_raw = []
test_data_raw = []

##Training mode
if mode == 0:
    for index, fileN in enumerate(fileNames) :
        train_data_raw.append(util.readFile(dataFolder, fileN[0], 1, 5500))
        #test_data_raw.append([])
        #test_data_raw[index].append(util.readFile(dataFolder, fileN[0], 5000, 5500))

    for index, data in enumerate(train_data_raw):
        util.printString('File: ' + fileNames[index][0], "File read:")  
        util.printList(data, 'Data read:')  
    
for index, data in enumerate(train_data_raw):
    train_data_raw[index] = util.cleanUpData(data)   

for i, d in enumerate(test_data_raw):
    for index, data in enumerate(d):
        test_data_raw[i][index] = util.cleanUpData(data)     

for index, data in enumerate(train_data_raw):
    util.printString('Training Dataset: ' + fileNames[index][0].split('.')[0]) 
    util.printList(data, 'After cleanup:')     

#for i, d in enumerate(test_data_raw):
    #for index, data in enumerate(d):
        #util.printString('Testing Dataset ',index,' of: ' + fileNames[index][0].split('.')[0]) 
        #util.printList(data, 'After cleanup:')  

target_names = ['Negative', 'Neutral', 'Positive']
classification.init()

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