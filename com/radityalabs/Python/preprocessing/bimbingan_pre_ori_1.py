# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
from nltk.corpus import movie_reviews
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk import NaiveBayesClassifier as nbc
from nltk.tokenize import word_tokenize
import random

random.seed(1)

train = [
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
    ('The beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]

sentence = "The beer was amazing. But i am not enjoy it"

# 1. The best one
# def end_word_extractor(document):
#     tokens = document.split()
#     first_word, last_word = tokens[0], tokens[-1]
#     feats = {}
#     feats["first({0})".format(first_word)] = True
#     feats["last({0})".format(last_word)] = False
#     return feats
# cl2 = NaiveBayesClassifier(train, feature_extractor=end_word_extractor)
# blob = TextBlob(sentence, classifier=cl2)
# print(blob.classify())
# print("Accuracy: {0}".format(cl2.accuracy(test)))

# 2. Second
# cl = NaiveBayesClassifier(train)
# print(cl.classify(sentence))
# print("Accuracy: {0}".format(cl.accuracy(test)))
# cl.show_informative_features(5)

# 3. Thrid
# vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in train]))
# feature_set = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag) for sentence, tag in train]
# classifier = nbc.train(feature_set)
# featurized_test_sentence = {i: (i in word_tokenize(sentence.lower())) for i in vocabulary}
# print("test_sent:", sentence)
# print("tag:", classifier.classify(featurized_test_sentence))