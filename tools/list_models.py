import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('GEMINI_API_KEY')
endpoints = [
    'https://generativelanguage.googleapis.com/v1/models',
    'https://generativelanguage.googleapis.com/v1beta1/models',
    'https://generativelanguage.googleapis.com/v1beta2/models',
]
for e in endpoints:
    try:
        r = requests.get(e, params={'key': key} if key else None, timeout=20)
        print('ENDPOINT:', e)
        print('STATUS:', r.status_code)
        print(r.text[:4000])
        print('\n' + '='*60 + '\n')
    except Exception as ex:
        print('ENDPOINT:', e)
        print('ERROR:', ex)
        print('\n' + '='*60 + '\n')
