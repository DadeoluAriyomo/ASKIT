import requests, json

url = 'http://127.0.0.1:5000/ask'
payload = {'question': 'Hello from local test: are you working?'}
try:
    r = requests.post(url, json=payload, timeout=30)
    print('STATUS:', r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print('RESPONSE_TEXT:', r.text)
except Exception as e:
    print('ERROR:', e)
