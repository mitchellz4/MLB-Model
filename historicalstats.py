import requests
from bs4 import BeautifulSoup as bs
import pickle

game_links = []
for current_year in range(2016,2022):
    url = f"https://www.baseball-reference.com/leagues/MLB/{current_year}-schedule.shtml"
    resp = requests.get(url)
    soup=bs(resp.text)
    games = soup.findAll('a',string='Boxscore')
    game_links.extend([x['href'] for x in games])
print("Number of games to download: ", len(game_links))



def get_game_summary(soup, game_id):
    game = {'game_id': game_id}
    scorebox = soup.find('div', {'class':'scorebox'})
    teams = scorebox.findAll('strong')
    if len(teams) == 0:  # if no team names are found, return None
        return None
    game['away_team_abbr'] = teams[0].text.split()[-1]
    game['home_team_abbr'] = teams[1].text.split()[-1]
    meta = scorebox.find('div', {'class':'scorebox_meta'}).findAll('div')
    game['date'] = meta[0].text.strip()
    game['time'] = meta[1].text.strip()
    game['location'] = meta[2].text.strip()
    return game

def get_table_summary(soup, table_no):
    stats_tables = soup.findAll('table', {'class':'stats_table'})
    t = stats_tables[table_no].find('tfoot')
    summary = {x['data-stat']:x.text.strip() for x in t.findAll('td')}
    return summary


def get_pitcher_data(soup, table_no):
    stats_tables = soup.findAll('table', {'class':'stats_table'})
    t = stats_tables[table_no]
    data = []
    rows = t.findAll('tr')[1:-1] # not the header and footer rows
    for r in rows:
        summary = {x['data-stat']:x.text.strip() for x in r.findAll('td')}
        summary['name'] = r.find('th',{'data-stat':'player'}).find('a')['href'].split('/')[-1][:-6].strip()
        data.append(summary)
    return data
   
def process_link(url):
    resp = requests.get(url)
    game_id = url.split('/')[-1][:-6]

    # strange preprocessing routine
    uncommented_html = ''
    for h in resp.text.split('\n'):
        if '<!--     <div' in h: continue
        if h.strip() == '<!--': continue
        if h.strip() == '-->': continue
        uncommented_html += h + '\n'

    soup = bs(uncommented_html)
    data = {
        'game': get_game_summary(soup, game_id),
        'away_batting': get_table_summary(soup, 1),
        'home_batting':get_table_summary(soup, 2),
        'away_pitching':get_table_summary(soup, 3),
        'home_pitching':get_table_summary(soup, 4),
        'away_pitchers': get_pitcher_data(soup, 3),
        'home_pitchers': get_pitcher_data(soup, 4)
    }
    return data     
import datetime as dt
game_data = []
for link in game_links:
    url = 'https://www.baseball-reference.com' + link
    game_data.append(process_link(url))
    if len(game_data)%1000==0: print(dt.datetime.now().time(), len(game_data))
import pickle
pickle.dump(game_data, open('game_data.pkl', 'wb'))