from textblob.classifiers import NaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from sklearn.externals import joblib
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from itertools import chain

training_data = [
    ('I love this sandwich.', 'pos'),
    ('This is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('This is my best work.', 'pos'),
    ("What an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('He is my sworn enemy!', 'neg'),
    ('My boss is horrible.', 'neg')
]

test = [
    ('the beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))
feature_set = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag) for sentence, tag in
               training_data]

classifier = NaiveBayesClassifier(training_data)
test_sentence = "This beer is good"
featurized_test_sentence = {i: (i in word_tokenize(test_sentence.lower())) for i in vocabulary}

print("test_sent:", test_sentence)
print("tag:", classifier.classify(featurized_test_sentence))
print("Accuracy: {0}".format(classifier.accuracy(test)))


# text = "Hi, I want to get the bigram list of this string"
# for item in nltk.bigrams (text.split()): print(' '.join(item))

# def bigramReturner(tweetString):
#     tweetString = tweetString.lower()
#     tweetString = removePunctuation(tweetString)
#     bigramFeatureVector = []
#     for item in nltk.bigrams(tweetString.split()):
#         bigramFeatureVector.append(' '.join(item))
#     return bigramFeatureVector
