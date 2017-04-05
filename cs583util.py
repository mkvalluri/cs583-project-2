##File: cs583util.py
##Description: Contains utilities which are used across the applicatin
##Author(s): Murali Krishna Valluri, Spoorthi Pendyala
##Date: 06/29/2017

import itertools
import re

debug_mode = 0
stop_words = []

##Opens csv file and returns the content of it in the form of string[]
def readFile(rootPath, fileName, firstIndex, lastIndex):
    lines = []
    with open(rootPath + fileName, 'r') as f:
        for line in itertools.islice(f, firstIndex, lastIndex):
            lines.append(line)
    
    return lines

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
    with open('data/stopwords.txt') as f:
        stop_words = f.read().split()
    
    for line in data:
        try:
            tempClass = line.split(',', 1)[0]
            tempLine = line.split(',', 1)[1]

            ##Convert everything to lowercase.
            tempLine = tempLine.lower()

            ##Remove htmltags.
            tempLine = re.sub(r'<[^>]+>', '', tempLine)

            #remove hyperlinks
            tempLine = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', tempLine)
            
            #Remove single quotes
            tempLine = re.sub("'", "", tempLine)
            
            #Remove double quotes
            tempLine = re.sub("\"", "", tempLine)
            
            #Remove citations
            tempLine = re.sub(r'@', '', tempLine)
            
            #Remove hashtags
            tempLine = re.sub(r'#', '', tempLine)

            ##Strip starting and ending spaces.
            tempLine = re.sub( '\s+', ' ', tempLine).strip()    

            tempLine = tempLine.encode('utf-8').strip()

            tempLine = removeStopWords(tempLine, stop_words)

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