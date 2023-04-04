import requests
import datetime as dt
import pickle
from bs4 import BeautifulSoup as bs

# Load the dates list from the pickle file
with open('mlb_dates.pickle', 'rb') as f:
    dates = pickle.load(f)

game_data = []
for d in dates:
    # get the web page with game data on it
    game_day = d.strftime('%Y-%m-%d')
    url = f'https://www.covers.com/Sports/MLB/Matchups?selectedDate={game_day}'
    resp = requests.get(url)

    # parse the games
    scraped_games = bs(resp.text, 'html.parser').findAll('div', {'class': 'cmg_matchup_game_box'})
    for g in scraped_games:
        game = {}
        game['home_moneyline'] = g['data-game-odd']
        game['date'] = g['data-game-date']
        try:
            game['home_score'] = g.find('div', {'class': 'cmg_matchup_list_score_home'}).text.strip()
            game['away_score'] = g.find('div', {'class': 'cmg_matchup_list_score_away'}).text.strip()
        except:
            game['home_score'] = ''
            game['away_score'] = ''

        game_data.append(game)
        if len(game_data) % 500 == 0:
            # show progress
            print(dt.datetime.now(), game_day, len(game_data))

print("Done! Games downloaded:", len(game_data))

import pickle
pickle.dump(game_data, open('covers_data.pkl','wb'))