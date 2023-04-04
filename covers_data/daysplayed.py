import requests
import re
import datetime as dt
import pickle

start_year = 2021
end_year = 2022
dates = []

for year in range(start_year, end_year + 1):
    url = f'https://www.baseball-reference.com/leagues/MLB/{year}-schedule.shtml'
    resp = requests.get(url)
    # All the H3 tags contain day names
    days = re.findall("<h3>(.*" + str(year) + ")</h3>", resp.text)
    year_dates = [dt.datetime.strptime(d,"%A, %B %d, %Y") for d in days]
    dates += year_dates

print("Number of days MLB was played from 2015 to 2022:", len(dates))
with open('mlb_dates.pickle', 'wb') as f:
    pickle.dump(dates, f)