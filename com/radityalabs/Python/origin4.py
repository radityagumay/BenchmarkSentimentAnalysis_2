#http://streamhacker.com/2010/05/17/text-classification-sentiment-analysis-precision-recall/
import collections
import nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

def word_feats(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4

trainfeats = negfeats[:int(negcutoff)] + posfeats[:int(poscutoff)]
testfeats = negfeats[int(negcutoff):] + posfeats[int(poscutoff):]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print('pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos']))
print('pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos']))
print('pos F-measure:', nltk.metrics.f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg']))
print('neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg']))
print('neg F-measure:', nltk.metrics.f_measure(refsets['neg'], testsets['neg']))
