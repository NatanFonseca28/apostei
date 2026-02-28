import sqlite3, requests, os, difflib, json
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get('ODDS_API_KEY')

resp = requests.get(
    'https://api.the-odds-api.com/v4/sports/soccer_brazil_campeonato/odds', 
    params={'apiKey': key, 'regions': 'eu', 'markets': 'h2h'}
)

if resp.status_code != 200:
    print("API Auth Error", resp.text)
    exit(1)

api_teams = set()
for e in resp.json():
    api_teams.add(e['home_team'])
    api_teams.add(e['away_team'])

conn = sqlite3.connect('flashscore_data.db')
db_teams = [row[0] for row in conn.execute('SELECT DISTINCT time_casa FROM flashscore_matches')]

print("Mapeamento Sugerido:\n_API_TEAM_MAP = {")
for t in sorted(api_teams):
    matches = difflib.get_close_matches(t, db_teams, n=1, cutoff=0.4)
    best_match = matches[0] if matches else "(NOT FOUND)"
    print(f'    "{t}": "{best_match}",')
print("}")
