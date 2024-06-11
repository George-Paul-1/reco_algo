import json 
import urllib
import requests 
import pandas as pd

url = 'https://parseapi.back4app.com/classes/Country?limit=10&include=continent&keys=name,emoji,capital,continent,continent.name,native,currency'
headers = {
    'X-Parse-Application-Id': 'mxsebv4KoWIGkRntXwyzg6c6DhKWQuit8Ry9sHja', # This is the fake app's application id
    'X-Parse-Master-Key': 'TpO0j3lG2PmEVMXlKYQACoOXKQrL3lwM0HwR9dbH' # This is the fake app's readonly master key
}

data = json.loads(requests.get(url, headers=headers).content.decode('utf-8')) # Here you have the data that you need

norm = pd.json_normalize(data, record_path = ['results'], record_prefix='dbscan_') 
norm.to_csv('norm.csv', index=False)

metadata = pd.read_csv('norm.csv')
print(metadata)

