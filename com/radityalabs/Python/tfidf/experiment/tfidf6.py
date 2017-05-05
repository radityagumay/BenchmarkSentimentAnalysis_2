# http://stackoverflow.com/questions/27446204/how-do-i-use-use-the-tfidf-calculating-functions-in-scikit-learn
# http://www.markhneedham.com/blog/2015/02/15/pythonscikit-learn-calculating-tfidf-on-how-i-met-your-mother-transcripts/
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import os

# SentenceId,EpisodeId,Season,Episode,Sentence
path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/")

episodes = defaultdict(list)
with open(path + "Python/tfidf/sentence.csv", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter=',')
    for row in reader:
        episodes[row[1]].append(row[4])

for episode_id, text in episodes.items():
    episodes[episode_id] = "".join(text)

corpus = []
for id, episode in sorted(episodes.items(), key=lambda t: int(t[0])):
    corpus.append(episode)

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(corpus)
feature_names = tf.get_feature_names()

dense = tfidf_matrix.todense()
print(dense)