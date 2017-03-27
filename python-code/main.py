import nltk
import re
import operator
import string
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams
from collections import Counter

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

line = ''
labels = []
features = []
with open('data/obama.csv', 'r') as f:
    count_all = Counter()  
    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + [u'rt', u'via', u'RT', u'Rt', u'Via']     
    #print (stop) 
    
    #for line in f:
    for line in itertools.islice(f, 3, 10):
        try:
            tempData = line.split(',', 1)
            labels.append(tempData[0])
            tempLine = tempData[1]

            #remove hyperlinks
            temp = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', tempLine)
            #Remove quotes
            temp = re.sub(r'&amp;quot;|&amp;amp', '', temp)
            #Remove citations
            temp = re.sub(r'@[a-zA-Z0-9]*', '', temp)
            #Remove tickers
            temp = re.sub(r'\$[a-zA-Z0-9]*', '', temp)
            #Remove html tags
            temp = re.sub(r'<[^>]+>', '', temp)
            #Remove hashtags
            temp = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", '', temp)
            #Remove leading spaces
            temp = temp.strip()
            #To lowercase
            temp = temp.lower()

            terms_all = [term for term in preprocess(temp) if term not in stop]
            features.append(terms_all)
            #terms_bigram = bigrams(terms_all)
            #print(unicode(line, errors='ignore'))        
            #print(unicode(temp, errors='ignore'))
            
            print(terms_all, tempData[0])
            #count_all.update(terms_all)
        except IndexError:
            print line
    
#print(count_all.most_common(500))
    
####
#features = np.random.rand(100, 10)
#labels = np.zeros(100)
#features[50:] += .5
#labels[50:] = 1
#for index,item in enumerate(features):
#    print(features[index] , labels[index])
#print(len(features))
#print(len(labels))
#learner = milk.defaultclassifier()
#model = learner.train(features, labels)
#print(model)
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(features)
#print(X_train_counts.shape)
#clf = MultinomialNB()
#clf = clf.fit(features, labels)
#print(clf.predict(['hate', 'obama']))


#import numpy
#from sklearn.feature_extraction.text import CountVectorizer

#count_vectorizer = CountVectorizer()
#counts = count_vectorizer.fit_transform(features)