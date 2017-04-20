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

    def __init__(self, index, id, text):
        try:
            label = request_nltk_api(index, text)
            if label == "":
                return
            else:
                insert_label(id, label)
        except:
            print("Error")
            training_benchmark(index)

def request_nltk_api(index, text):
    try:
        payload = {
            'language': 'english',
            'text': text
        }
        request = requests.post(url, data=payload)
        response = json.loads(request.text)
    except requests.exceptions.HTTPError as e:
        print("Error: " + str(e))
        training_benchmark(index)
        return None
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

def training_benchmark(ind):
    index = 0 if ind == "" else ind
    for data in datas:
        index += 1
        print("index:", index)
        if index > 2377:
            Benchmark(index, data[0], data[1])
    close_connection()

training_benchmark(0)