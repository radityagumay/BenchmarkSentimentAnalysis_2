from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
# clf = svm.SVC()
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# joblib.dump(clf, 'sklearn-joblib.pkl')

# clf = joblib.load('sklearn-joblib.pkl')
#
# iris = datasets.load_iris()
# x, y = iris.data, iris.target
# print(clf.fit(x, y))

from textblob.classifiers import NaiveBayesClassifier

# train = [
#     ('I love this sandwich.', 'pos'),
#     ('This is an amazing place!', 'pos'),
#     ('I feel very good about these beers.', 'pos'),
#     ('This is my best work.', 'pos'),
#     ("What an awesome view", 'pos'),
#     ('I do not like this restaurant', 'neg'),
#     ('I am tired of this stuff.', 'neg'),
#     ("I can't deal with this", 'neg'),
#     ('He is my sworn enemy!', 'neg'),
#     ('My boss is horrible.', 'neg')
# ]
#
# cl = NaiveBayesClassifier(train)
# joblib.dump(cl, 'sklearn-joblib.pkl')

sentence = "The beer was amazing. But, I do not enjoy it"
cl = joblib.load('sklearn-joblib.pkl')
print(cl.classify(sentence))  # "neg"