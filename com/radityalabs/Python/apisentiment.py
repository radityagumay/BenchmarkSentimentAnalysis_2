import requests
import json
import pymysql

# variables
url = 'http://text-processing.com/api/sentiment/'
datas = []

db = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='google_play_crawler')

cursor = db.cursor()
cursor.execute("SELECT authorId, reviewBody FROM google_play_crawler.authors17")

def calculate():
    for data in cursor:
        datas.append((data[0], data[1]))

def close_connection():
    cursor.close()
    db.close()

calculate()

class Benchmark:
    def __init__(self):
        print("constructor")

    def __init__(self, id, text):
        label = request_nltk_api(text)
        insert_label(id, label)

def request_nltk_api(text):
    payload = {
        'language': 'english',
        'text': text
    }
    request = requests.post(url, data=payload)
    response = json.loads(request.text)
    negative_val = response['probability']['neg']
    positive_val = response['probability']['pos']
    label = response["label"]
    print("===========================")
    print("sentence:", text)
    print("negative:", negative_val)
    print("positive:", positive_val)
    print("label:", label)
    print("===========================")
    print("\n")
    return label

def insert_label(id, label):
    query = "UPDATE google_play_crawler.authors17 SET label='" + label + "' WHERE authorId='" + str(id) + "'"
    cursor.execute(query)
    db.commit()

def training_benchmark():
    for data in datas:
        Benchmark(data[0], data[1])
    close_connection()

training_benchmark()