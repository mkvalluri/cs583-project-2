##File: cs583util.py
##Description: Contains utilities which are used across the applicatin
##Author(s): Murali Krishna Valluri, Spoorthi Pendyala
##Date: 06/29/2017

import itertools
import re
import csv
import time

debug_mode = 0
stop_words = []

##Opens csv file and returns the content of it in the form of string[]
def readFile(rootPath, fileName, firstIndex, lastIndex):
    lines = []
    with open(rootPath + fileName, 'r') as f:
        for line in itertools.islice(f, firstIndex, lastIndex):
            lines.append(line)
    
    return lines

def writeFile(data):
    timestr = time.strftime("%Y%m%d%H%M%S")
    rootPath = 'output/'
    fileName = "Output" + timestr + ".csv"
    headerText = "Algorithm,Negative,,,Positive,,,Accuracy\n"
    headerText = headerText + ',Precision,Recall,F-Score,Precision,Recall,F-Score,\n'
    for idx, d in enumerate(data):
        data[idx] = headerText + d
        data[idx] = data[idx] + ',,,,,,,\n,,,,,,,\n,,,,,,,\n'

    with open(rootPath + fileName, 'w') as f:
        f.write(data[0] + data[1])

def printDescription(desc):
    if desc != '':
        print desc
        print '==============='
        

def printString(data, description = '', overrideDebug = 0):
    if debug_mode == 1 or overrideDebug == 1:
        printDescription(description)
        print data + '\n'

def printList(data, description = '', overrideDebug = 0):
    if debug_mode == 1 or overrideDebug == 1:
        printDescription(description)
        for line in data:
            print line
        print '\n'

def cleanUpData(data):
    classes = []
    lines = []

    slang_words = {}
    with open('data/acrynom.csv', mode='r') as infile:
        for l in infile:
            slang_words.update({l.split(',')[0]: l.split(',')[1]})
    
    smileys = {}
    with open('data/emoticonsWithPolarity.txt', mode='r') as infile:
        for l in infile:
            d1 = l.split('\t')
            d2 = d1[0].split(' ')
            for d3 in d2:
                smileys.update({d3: d1[1]})
    
    
    for line in data:
        try:
            tempClass = line.split(',', 1)[0]
            tempLine = line.split(',', 1)[1]

            printString(tempLine)

            ##Replace smileys.
            tempLine = replaceText(tempLine, smileys)

            ##Convert everything to lowercase.
            tempLine = tempLine.lower()

            ##Remove htmltags.
            tempLine = re.sub(r'<[^>]+>', '', tempLine)

            #Remove hyperlinks
            tempLine = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', tempLine)
            
            #Remove single quotes
            tempLine = re.sub("'", "", tempLine)
            
            #Remove double quotes
            tempLine = re.sub("\"", "", tempLine)
            
            #Remove citations
            tempLine = re.sub(r'@', '', tempLine)
            
            #Remove hashtags
            tempLine = re.sub(r'#', '', tempLine)

            #Remove slang words
            tempLine = replaceText(tempLine, slang_words)

            #Remove stop words
            #tempLine = removeStopWords(tempLine, stop_words)

            ##Strip starting and ending spaces.
            tempLine = re.sub( '\s+', ' ', tempLine).strip()    

            printString(tempLine)
            classes.append(tempClass)
            lines.append(tempLine)

        except IndexError:
            printString(line)
        
        except UnicodeDecodeError:
            printString(tempLine)

    returnData = []
    returnData.append(classes)
    returnData.append(lines)

    removeStopWords("This is a string", stop_words)

    return returnData

def removeStopWords(line, stop_words):
    for word in stop_words:
        line = line.replace(" " + word + " ", " ")
    return line

def replaceText(line, wordsDict):
    for key, value in wordsDict.iteritems():
        line = line.replace(" " + key + " ", " " + value + " ")
    return line