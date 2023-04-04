import pickle
df = pickle.load(open('dataframe.pkl', 'rb'))
from elote import EloCompetitor
ratings = {}
for x in df.home_team_abbr.unique():
    ratings[x]=EloCompetitor()
for x in df.away_team_abbr.unique():
    ratings[x]=EloCompetitor()

home_team_elo = []
away_team_elo = []
elo_exp = []

df = df.sort_values(by='date').reset_index(drop=True)
for i, r in df.iterrows():
    # get pre-game ratings
    elo_exp.append(ratings[r.home_team_abbr].expected_score(ratings[r.away_team_abbr]))
    home_team_elo.append(ratings[r.home_team_abbr].rating)
    away_team_elo.append(ratings[r.away_team_abbr].rating)
    # update ratings
    if r.home_team_win:
        ratings[r.home_team_abbr].beat(ratings[r.away_team_abbr])
    else:
        ratings[r.away_team_abbr].beat(ratings[r.home_team_abbr])

df['elo_exp'] = elo_exp
df['home_team_elo'] = home_team_elo
df['away_team_elo'] = away_team_elo

ratings = {}
for x in df.home_team_abbr.unique():
    ratings[x]=EloCompetitor()
    ratings[x]._k_score=16
for x in df.away_team_abbr.unique():
    ratings[x]=EloCompetitor()
    ratings[x]._k_score=16

home_team_elo = []
away_team_elo = []
elo_exp = []

df = df.sort_values(by='date').reset_index(drop=True)
for i, r in df.iterrows():
    # get pregame ratings
    elo_exp.append(ratings[r.home_team_abbr].expected_score(ratings[r.away_team_abbr]))
    home_team_elo.append(ratings[r.home_team_abbr].rating)
    away_team_elo.append(ratings[r.away_team_abbr].rating)
    # update ratings
    if r.home_team_win:
        ratings[r.home_team_abbr].beat(ratings[r.away_team_abbr])
    else:
        ratings[r.away_team_abbr].beat(ratings[r.home_team_abbr])

df['elo_slow_exp'] = elo_exp
df['home_team_elo_slow'] = home_team_elo
df['away_team_elo_slow'] = away_team_elo

from elote import GlickoCompetitor
ratings = {}
for x in df.home_team_abbr.unique():
    ratings[x]=GlickoCompetitor()
for x in df.away_team_abbr.unique():
    ratings[x]=GlickoCompetitor()

home_team_glick = []
away_team_glick = []
glick_exp = []

df = df.sort_values(by='date').reset_index(drop=True)
for i, r in df.iterrows():
    # get pregame ratings
    glick_exp.append(ratings[r.home_team_abbr].expected_score(ratings[r.away_team_abbr]))
    home_team_glick.append(ratings[r.home_team_abbr].rating)
    away_team_glick.append(ratings[r.away_team_abbr].rating)
    # update ratings
    if r.home_team_win:
        ratings[r.home_team_abbr].beat(ratings[r.away_team_abbr])
    else:
        ratings[r.away_team_abbr].beat(ratings[r.home_team_abbr])

df['glick_exp'] = glick_exp
df['home_team_glick'] = home_team_glick
df['away_team_glick'] = away_team_glick

from trueskill import Rating, quality, rate
ratings = {}
for x in df.home_team_abbr.unique():
    ratings[x]=Rating(25)
for x in df.away_team_abbr.unique():
    ratings[x]=Rating(25)
for x in df.home_pitcher.unique():
    ratings[x]=Rating(25)
for x in df.away_pitcher.unique():
    ratings[x]=Rating(25)

ts_quality = []
pitcher_ts_diff = []
team_ts_diff = []
home_pitcher_ts = []
away_pitcher_ts = []
home_team_ts = []
away_team_ts = []
df = df.sort_values(by='date').copy()
for i, r in df.iterrows():
    # get pre-match trueskill ratings from dict
    match = [(ratings[r.home_team_abbr], ratings[r.home_pitcher]),
            (ratings[r.away_team_abbr], ratings[r.away_pitcher])]
    ts_quality.append(quality(match))
    pitcher_ts_diff.append(ratings[r.home_pitcher].mu-ratings[r.away_pitcher].mu)
    team_ts_diff.append(ratings[r.home_team_abbr].mu-ratings[r.away_team_abbr].mu)
    home_pitcher_ts.append(ratings[r.home_pitcher].mu)
    away_pitcher_ts.append(ratings[r.away_pitcher].mu)
    home_team_ts.append(ratings[r.home_team_abbr].mu)
    away_team_ts.append(ratings[r.away_team_abbr].mu)

    if r.date < df.date.max():
        # update ratings dictionary with post-match ratings
        if r.home_team_win==1:
            match = [(ratings[r.home_team_abbr], ratings[r.home_pitcher]),
                     (ratings[r.away_team_abbr], ratings[r.away_pitcher])]
            [(ratings[r.home_team_abbr], ratings[r.home_pitcher]),
            (ratings[r.away_team_abbr], ratings[r.away_pitcher])] = rate(match)
        else:
            match = [(ratings[r.away_team_abbr], ratings[r.away_pitcher]),
                     (ratings[r.home_team_abbr], ratings[r.home_pitcher])]
            [(ratings[r.away_team_abbr], ratings[r.away_pitcher]),
            (ratings[r.home_team_abbr], ratings[r.home_pitcher])] = rate(match)

df['ts_game_quality'] = ts_quality
df['pitcher_ts_diff'] = pitcher_ts_diff
df['team_ts_diff'] = team_ts_diff
df['home_pitcher_ts'] = home_pitcher_ts
df['away_pitcher_ts'] = away_pitcher_ts
df['home_team_ts'] = home_team_ts
df['away_team_ts'] = away_team_ts

import pickle
pickle.dump(df, open('dataframe.pkl', 'wb'))