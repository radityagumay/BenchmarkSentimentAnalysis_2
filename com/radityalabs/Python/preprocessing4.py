from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
import progressbar
import asyncio

def word_feats(words):
    return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

def get_count():
    with open("corpus/polarity-data-negative.txt", "r") as file:
        return len(file.read().split("."))

@asyncio.coroutine
def load_positive():
    bar = progressbar.ProgressBar(maxval=get_count(), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    index = 0
    with open("corpus/polarity-data-negative.txt", "r") as file:
        data = file.read().split(".")
        for i in data:
            bar.update(index)
            index += 1
            data = i.split(" ")
            for j in data:
                if j not in negfeats:
                    negfeats.append((word_feats(word_tokenize(j)), 'neg'))
    bar.finish()
    print(negfeats)


asyncio.get_event_loop().run_until_complete(load_positive())