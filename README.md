# MLB-Model

This is an adapdation from rdpharr on github with updated code due to baseball reference changing their code.

https://rdpharr.github.io/project_notes/baseball/benchmark/webscraping/brier/accuracy/calibration/machine%20learning/2020/09/20/baseball_project.html

In the covers_data directory run the daysplayed.py file

In the same directory run the covers_data.py file
this will get the odds from 2021-2022 
covers got rid of odds from previous years so this is the best I could come up with


In the mlb_data directory run the historicalstats.py file

In the same directory run the dataprep.py file 

In the same directory run the powerrankings.py file

In the model directory you can run the hyperparam.py file to find better params or tweak them yourself

In the model directory run the xgb_model.py file 

In the main directory run the mlb_model_daily.ipynb file

This is a notebook for now until I can get it all tuned and looking nice in one main.py file

Have not added in the kelly criteria yet, as still trying to find a way to get odds for historical games to train model on that.



