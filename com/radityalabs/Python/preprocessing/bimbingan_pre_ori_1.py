# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
import random
from nltk.corpus import movie_reviews
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

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

cl = NaiveBayesClassifier(train)
print("I don't like their pizza.", cl.classify("I don't like their pizza."))  # "neg"
print("Accuracy: {0}".format(cl.accuracy(test)))
cl.show_informative_features(5)
