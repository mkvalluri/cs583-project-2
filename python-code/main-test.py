data = ['You won something!', 'are you asdasd', 'How are you', 'Murali won ssdf','how won asdsdf', 'Krishna won how is ur health', 'Your how name']
is_spam = [1, 1, 0, 1, 1, 0, 0]

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(data)
print vect.get_feature_names()

train_dtm_counts = vect.transform(data)
print train_dtm_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_dtm = tfidf_transformer.fit_transform(train_dtm_counts)
print train_dtm.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_dtm, is_spam)

data_new = ['how asdsdf', 'won asd dsf how isdfs ur health', 'Your f dsf dsf sdfname']
test_dtm = vect.transform(data_new)
actual = [1, 0, 0]
#predicted = clf.predict(test_dtm)
#target_names = ['No', 'Yes']
#for d, c in zip(data_new, predicted):
#    print ('%r => %s' % (d, target_names[c]))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, train_dtm, is_spam, cv=3)
print len(scores.split('\n'))
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#from sklearn import metrics
#print(metrics.classification_report(actual, predicted, target_names=target_names))

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect1', CountVectorizer()),
                     ('tfidf1', TfidfTransformer()),
                     ('clf1', MultinomialNB())])
text_clf = text_clf.fit(data, is_spam)
predicted = text_clf.predict(data_new)
target_names = ['No', 'Yes']
for d, c in zip(data_new, predicted):
    print ('%r => %s' % (d, target_names[c]))

from sklearn import metrics
print(metrics.classification_report(actual, predicted, target_names=target_names))