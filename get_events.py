import requests, os, json
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get('ODDS_API_KEY')

resp = requests.get(
    'https://api.the-odds-api.com/v4/sports/soccer_brazil_campeonato/odds', 
    params={'apiKey': key, 'regions': 'eu', 'markets': 'h2h'}
)

if resp.status_code == 200:
    events = []
    for e in resp.json()[:5]:
        events.append({
            "home": e["home_team"],
            "away": e["away_team"],
            "commence_time": e["commence_time"],
        })
    with open('api_events.json', 'w') as f:
        json.dump(events, f, indent=2)
else:
    with open('api_events.json', 'w') as f:
        f.write(resp.text)
