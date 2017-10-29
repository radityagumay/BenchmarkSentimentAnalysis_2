import csv
import os


class SentimentAnalysis(object):
    def __init__(self):
        self.path = os.path.expanduser("~/Python/SamplePython3/com/radityalabs/Python/positive-negative-data/")

    def load_positive_document(self):
        document = []
        with open(self.path + "positive-data.csv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                document.append((row, "positive"))
        return document

    def load_negative_document(self):
        document = []
        with open(self.path + "negative-data.csv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                document.append((row, "negative"))
        return document

    def retrieve_document(self):
        positive = self.load_positive_document()
        negative = self.load_negative_document()
        print(positive[:5])
        print(negative[:5])

    def export_to_csv(self):
        positive = self.load_positive_document()
        negative = self.load_negative_document()
        document = positive[:5] + negative[:5]
        n_doc = []
        for d in document:
            n_doc.append((d[0][0], d[1]))
        with open(self.path + "collection.csv", 'a') as outcsv:
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(['document', 'label'])
            for item in n_doc:
                writer.writerow([item[0], item[1]])

sentiment = SentimentAnalysis()
sentiment.export_to_csv()
