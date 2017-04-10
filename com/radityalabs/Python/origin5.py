# http://textblob.readthedocs.io/en/dev/classifiers.html
from textblob.classifiers import NaiveBayesClassifier

train = [
    ('I love this sandwich.', 'pos'),
    ('this is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('this is my best work.', 'pos'),
    ("what an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('he is my sworn enemy!', 'neg'),
    ('my boss is horrible.', 'neg')
]
test = [
    ('the beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]

sentence = "the beer was horrible."
cl = NaiveBayesClassifier(train)
#print("classifiy", cl.classify(sentence))
prob_dist = cl.prob_classify(sentence)
print(prob_dist.max())
print("probability positive:", round(prob_dist.prob("pos"), 2))
print("probability negative:", round(prob_dist.prob("neg"), 2))

# with open('train.json', 'r') as fp:
#     cl = NaiveBayesClassifier(fp, format="json")
