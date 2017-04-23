# http://stackoverflow.com/a/16344128/5435658
import pymysql
import requests
import json

# variables
base_url = 'http://text-processing.com/api/sentiment/'

def connection():
    conn = pymysql.connect(
        host='127.0.0.1',
        user='root', passwd='',
        db='google_play_crawler')
    cur = conn.cursor()
    cur.execute("SELECT authorId, authorName, authorDetailLink, reviewBody from authors17 limit 1, 4480")
    return cur, conn

conn_polarity = pymysql.connect(
    host='127.0.0.1',
    user='root', passwd='',
    db='sentiment_analysis')
cur_polarity = conn_polarity.cursor()

def close_connection(conn, cur):
    conn.close
    cur.close
    conn_polarity.close()
    cur_polarity.close()

def insert_new_data_twitter(index, name, googleId, review, positive, negative, neutral, polarity, label):
    insert_query = '"' + \
                   name + '", "' + \
                   str(googleId) + '", "' + \
                   review + '", ' + \
                   str(positive) + ', ' + \
                   str(negative) + ', ' + \
                   str(neutral) + ', ' + \
                   str(polarity) + ', "' + \
                   label + '"'
    print("============ INSERT ============ ", index)
    print(insert_query)
    query = "INSERT INTO sentiment_analysis.review_label_benchmark_with_polarity (authorName, googleId, reviewBody, positive, negative, neutral, polarity, label) values (" + insert_query + ")"
    cur_polarity.execute(query)
    conn_polarity.commit()
    print("============ DONE ============")
    print("\n")

def request(data, index):
    review = data[3]
    review = review.replace('"', '\\"')
    payload = {
        'language': 'english',
        'text': review
    }
    request = requests.post(base_url, data=payload, )
    response = json.loads(request.text)
    # id = data[0]
    name = data[1]
    googleId = get_id(data[2])
    negative = response['probability']['neg']
    neutral = response['probability']['neutral']
    positive = response['probability']['pos']
    polarity = 1 - neutral
    label = response['label']
    insert_new_data_twitter(index, name, googleId, review, positive, negative, neutral, polarity, label)

def get_id(value):
    data = value
    data = data.split("/")
    if data[5] == "apps":
        return value
    elif data[5] == "people":
        data = data[6].split("=")
        return data[1]
    else:
        return value

last_index = 1302

def select_new_data_twitter():
    cur, conn = connection()
    index = 0
    for data in cur:
        index += 1
        if index > last_index:
            request(data, index)
    close_connection(conn, cur)

print("start script", select_new_data_twitter())

# data = "https://play.google.com//store/people/details?id=117141524971167116964"
# data = data.split("/")
# data = data[6].split("=")
# data = int(data[1])
#
# data2 = "https://play.google.com//store/apps/details?id=com.twitter.android&reviewId=bGc6QU9xcFRPR0NmMVBWZk9YUEtOVW5JbDduM2hCS3ZzZ1NTRWVDYjZCU2VpTU1GWUhtRTBNOU9hM1cxZW5qRElZR1JHa0Vfd3hzVTBUV19obnFHZDF0UHc"
# data2 = data2.split("/")
# data2 = data2[6].split("=")
#
# print(isinstance(data, int))
# print(data2[1])
