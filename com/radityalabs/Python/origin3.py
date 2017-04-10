from nltk import NaiveBayesClassifier as nbc
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk.corpus import movie_reviews

training_data = [('I love this sandwich.', 'pos'),
                 ('This is an amazing place!', 'pos'),
                 ('I feel very good about these beers.', 'pos'),
                 ('This is my best work.', 'pos'),
                 ("What an awesome view", 'pos'),
                 ('I do not like this restaurant', 'neg'),
                 ('I am tired of this stuff.', 'neg'),
                 ("I can't deal with this", 'neg'),
                 ('He is my sworn enemy!', 'neg'),
                 ('My boss is horrible.', 'neg')]

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

def build_training_data():
    new_training_data = []
    for n in negids:
        sentence = movie_reviews.raw(fileids=[n])
        new_training_data.append((sentence, 'neg'))
    for p in posids:
        sentence = movie_reviews.raw(fileids=[p])
        new_training_data.append((sentence, 'pos'))
    return build_training_data()

new_training_data = build_training_data()

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))
feature_set = [({i: (i in word_tokenize(sentence.lower())) for i in vocabulary}, tag) for sentence, tag in new_training_data]
classifier = nbc.train(feature_set)

test_sentence = "This is the bad band I've ever heard!"
featurized_test_sentence = {i: (i in word_tokenize(test_sentence.lower())) for i in vocabulary}

print("test_sent:", test_sentence)
print("tag:", classifier.classify(featurized_test_sentence))
