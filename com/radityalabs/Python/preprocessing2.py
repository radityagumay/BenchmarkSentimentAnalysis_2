import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize


def word_feats(words):
    return dict([(word, True) for word in words])


negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

vocabulary = {}

for i in negids:
    for j in movie_reviews.words(fileids=[i]):
        vocabulary[j] = j

for i in posids:
    for j in movie_reviews.words(fileids=[i]):
        vocabulary[j] = j

negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4

trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
classifier.show_most_informative_features()

test_sentence = "This best movie ever!!"
featurized_test_sentence = {i: (i in word_tokenize(test_sentence)) for i in vocabulary}

print("example:", test_sentence)
print("label:", classifier.classify(featurized_test_sentence))
