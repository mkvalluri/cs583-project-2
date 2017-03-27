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

data_new = ['how won asdsdf', 'Krishna won how is ur health', 'Your how name']
test_dtm = vect.transform(data_new)
actual = [1, 0, 0]
predicted = clf.predict(test_dtm)
target_names = ['No', 'Yes']
for d, c in zip(data_new, predicted):
    print ('%r => %s' % (d, target_names[c]))

from sklearn import metrics
print(metrics.classification_report(actual, predicted, target_names=target_names))