import pickle
game_data = pickle.load(open('game_data.pkl', 'rb'))
import pandas as pd

games = []
batting = []
pitching = []
pitchers = []

for g in game_data:
    game_summary = g['game']

    # fix date
   # game_summary['date'] = game_summary['date'] + " " + game_summary['start_time']
    #del game_summary['start_time']

    # get starting pitchers
    game_summary['home_pitcher'] = g['home_pitchers'][0]['name']
    game_summary['away_pitcher'] = g['away_pitchers'][0]['name']

    # this is the field we'll train our model to predict
    game_summary['home_team_win'] = int(g['home_batting']['R'])>int(g['away_batting']['R'])
    games.append(game_summary)

    # add all stats to appropriate lists
    target_pairs = [
        ('away_batting', batting),
        ('home_batting', batting),
        ('away_pitching', pitching),
        ('home_pitching', pitching),
        ('away_pitchers', pitchers),
        ('home_pitchers', pitchers)
    ]
    for key, d in target_pairs:
        if isinstance(g[key], list): # pitchers
            for x in g[key]:
                if 'home' in key:
                    x['is_home_team'] = True
                    x['team'] = g['game']['home_team_abbr']
                else:
                    x['is_home_team'] = False
                    x['team'] = g['game']['away_team_abbr']
                x['game_id'] = g['game']['game_id']
                d.append(x)
        else: #batting, pitching
            x = g[key]
            if 'home' in key:
                x['is_home_team'] = True
                x['team'] = g['game']['home_team_abbr']
                x['spread'] = int(g[key]['R']) - int(g[key.replace('home','away')]['R'])
            else:
                x['is_home_team'] = False
                x['team'] = g['game']['away_team_abbr']
                x['spread'] = int(g[key]['R']) - int(g[key.replace('away','home')]['R'])
            x['game_id'] = g['game']['game_id']
            d.append(x)
len(games), len(batting), len(pitching), len(pitchers)
game_df = pd.DataFrame(games)
#TODO: fix games that were rescheduled which become NaT after this next command
game_df['date'] = pd.to_datetime(game_df['date'], errors='coerce')
game_df = game_df[~game_df['game_id'].str.contains('allstar')].copy() #don't care about allstar games
game_df.head()

batting_df = pd.DataFrame(batting)
for k in batting_df.keys():
    if any(x in k for x in ['team','game_id', 'home_away']): continue
    batting_df[k] =pd.to_numeric(batting_df[k],errors='coerce', downcast='float')
batting_df.drop(columns=['details'], inplace=True)
batting_df.head()

pitching_df = pd.DataFrame(pitching)
for k in pitching_df.keys():
    if any(x in k for x in ['team','game_id', 'home_away']): continue
    pitching_df[k] =pd.to_numeric(pitching_df[k],errors='coerce', downcast='float')
pitching_df.head()

pitcher_df = pd.DataFrame(pitchers)
for k in pitcher_df.keys():
    if any(x in k for x in ['team','name','game_id', 'home_away']): continue
    pitcher_df[k] =pd.to_numeric(pitcher_df[k],errors='coerce', downcast='float')
# filter the pitcher performances to just the starting pitcher
pitcher_df = pitcher_df[~pitcher_df['game_score'].isna()].copy().reset_index(drop=True)
pitcher_df.drop(columns=[x for x in pitcher_df.keys() if 'inherited' in x], inplace=True)
pitcher_df.head()


##
import numpy as np

def add_rolling(period, df, stat_columns):
    for s in stat_columns:
        if 'object' in str(df[s].dtype): continue
        df[s+'_'+str(period)+'_Avg'] = df.groupby('team')[s].apply(lambda x:x.rolling(period).mean())
        df[s+'_'+str(period)+'_Std'] = df.groupby('team')[s].apply(lambda x:x.rolling(period).std())
        df[s+'_'+str(period)+'_Skew'] = df.groupby('team')[s].apply(lambda x:x.rolling(period).skew())
    return df

def get_diff_df(df, name, is_pitcher=False):
    #runs for each of the stat dataframes, returns the difference in stats

    #set up dataframe with time index
    df['date'] = pd.to_datetime(df['game_id'].str[3:-1], format="%Y%m%d")
    df = df.sort_values(by='date').copy()
    newindex = df.groupby('date')['date']\
             .apply(lambda x: x + np.arange(x.size).astype(np.timedelta64(0,'s')))
    df = df.set_index(newindex).sort_index()

    # get stat columns
    stat_cols = [x for x in df.columns if 'int' in str(df[x].dtype)]
    stat_cols.extend([x for x in df.columns if 'float' in str(df[x].dtype)])

    #add lags
    df = add_rolling('5d', df, stat_cols) # this game series
    df = add_rolling('10d', df, stat_cols)
    df = add_rolling('45d', df, stat_cols)
    df = add_rolling('180d', df, stat_cols) # this season
    df = add_rolling('730d', df, stat_cols) # 2 years

    # reset stat columns to just the lags (removing the original stats)
    df.drop(columns=stat_cols, inplace=True)
    stat_cols = [x for x in df.columns if 'int' in str(df[x].dtype)]
    stat_cols.extend([x for x in df.columns if 'float' in str(df[x].dtype)])

    # shift results so that each row is  a pregame stat
    df = df.reset_index(drop=True)
    df = df.sort_values(by='date')
    for s in stat_cols:
        if is_pitcher:
            df[s] = df.groupby('name')[s].shift(1)
        else:
            df[s] = df.groupby('team')[s].shift(1)

    # calculate differences in pregame stats from home vs. away teams
    away_df = df[~df['is_home_team']].copy()
    away_df = away_df.set_index('game_id')
    away_df = away_df[stat_cols]

    home_df = df[df['is_home_team']].copy()
    home_df = home_df.set_index('game_id')
    home_df = home_df[stat_cols]

    diff_df = home_df.subtract(away_df, fill_value=0)
    diff_df = diff_df.reset_index()

    # clean column names
    for s in stat_cols:
        diff_df[name + "_" + s] = diff_df[s]
        diff_df.drop(columns=s, inplace=True)

    return diff_df
df = game_df

df = pd.merge(left=df, right = get_diff_df(batting_df, 'batting'),
               on = 'game_id', how='left')
print(df.shape)

df = pd.merge(left=df, right = get_diff_df(pitching_df, 'pitching'),
               on = 'game_id', how='left')
print(df.shape)

df = pd.merge(left=df, right = get_diff_df(pitcher_df, 'pitcher',is_pitcher=True),
               on = 'game_id', how='left')
df.shape
pitcher_df = pd.DataFrame(pitchers) # old version was filtered to just starters
dates = pitcher_df['game_id'].str[3:-1]
pitcher_df['date'] = pd.to_datetime(dates,format='%Y%m%d', errors='coerce')
pitcher_df['rest'] = pitcher_df.groupby('name')['date'].diff().dt.days

# merge into main dataframe
# filter the pitcher performances to just the starting pitcher
pitcher_df = pitcher_df[~pitcher_df['game_score'].isna()].copy().reset_index(drop=True)
home_pitchers = pitcher_df[pitcher_df['is_home_team']].copy().reset_index(drop=True)
df = pd.merge(left=df, right=home_pitchers[['game_id','name', 'rest']],
              left_on=['game_id','home_pitcher'],
              right_on=['game_id','name'],
              how='left')
df.rename(columns={'rest':'home_pitcher_rest'}, inplace=True)

away_pitchers = pitcher_df[~pitcher_df['is_home_team']].copy().reset_index(drop=True)
df = pd.merge(left=df, right=away_pitchers[['game_id','name','rest']],
              left_on=['game_id','away_pitcher'],
              right_on=['game_id','name'],
              how='left')
df.rename(columns={'rest':'away_pitcher_rest'}, inplace=True)

df['rest_diff'] = df['home_pitcher_rest']-df['away_pitcher_rest']
df.dropna(subset=['date'], inplace=True)
df['season'] = df['date'].dt.year
df['month']=df['date'].dt.month
df['week']=df['date'].dt.isocalendar().week.astype('int')
df['dow']=df['date'].dt.weekday
df['date'] = (pd.to_datetime(df['date']) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') #epoch time

df.shape

import pickle
pickle.dump(df, open('dataframe.pkl', 'wb'))

