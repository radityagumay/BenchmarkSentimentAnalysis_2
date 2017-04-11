import requests
import json

url = 'http://text-processing.com/api/sentiment/'
payload = {
    'language': 'english',
    'text': "Still auto refreshes, scrolls to top on it's own, so annoying! "
}
request = requests.post(url, data=payload)
response = json.loads(request.text)
print(response['probability']['neg'])