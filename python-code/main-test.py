data = ['You won something!', 'How are you', 'Murali won ssdf']
is_spam = [1, 0, 1]

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(data)
print vect.get_feature_names()

train_dtm = vect.transform(data)
print train_dtm.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_dtm, is_spam)

data_new = ['how won asdsdf', 'won is ur health']
test_dtm = vect.transform(data_new)

predicted = clf.predict(test_dtm)
target_names = ['Yes', 'No']
for d, c in zip(data_new, predicted):
    print ('%r => %s' % (d, is_spam[c]))

from sklearn import metrics
print (metrics.classification_report(data_new, predicted, target_names=target_names))