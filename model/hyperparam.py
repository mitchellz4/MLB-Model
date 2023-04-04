
import pickle
df = pickle.load(open('dataframe.pkl', 'rb'))
df.shape

encode_me = [x for x in df.keys() if 'object' in str(df[x].dtype)]
for x in encode_me:
    df[x] = df.groupby(x, group_keys=False)['home_team_win'].apply(lambda x:x.rolling(180).mean()).shift(1)

df = df.sort_values(by='date').copy().reset_index(drop=True)
X = df.drop(columns=['home_team_win', 'game_id'])
y = df.home_team_win

X_train = X[:-7389]
y_train = y[:-7389]
X_valid = X[-7389:-2000]
y_valid = y[-7389:-2000]
X_test = X[-2000:]
y_test = y[-2000:]

from hyperopt import fmin, tpe, hp, Trials
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss

def get_xgb_model(params):
    # comment the next 2 lines out if you don't have gpu
    params['gpu_id'] = 0
    params['tree_method'] = 'gpu_hist'
    params['seed']=13

    gbm = xgb.XGBClassifier(**params,n_estimators=999)
    model = gbm.fit(X_train, y_train,
                    verbose=False,
                    eval_set = [[X_train, y_train],
                              [X_valid, y_valid]],
                    eval_metric='logloss',
                    early_stopping_rounds=15)
    return model

def xgb_objective(params):
    params['max_depth']=int(params['max_depth'])
    model = get_xgb_model(params)
    xgb_test_proba = model.predict_proba(X_valid)[:,1]
    score = brier_score_loss(y_valid, xgb_test_proba)
    return(score)

trials = Trials() # recorder for our results

def get_xgbparams(space, evals=15):
    params = fmin(xgb_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials)
    params['max_depth']=int(params['max_depth'])
    return params

import numpy as np
hyperopt_runs = 500

space = {
    'max_depth':  hp.quniform('max_depth', 1, 8, 1),
    'min_child_weight': hp.quniform('min_child_weight', 3, 15, 1),
    'learning_rate': hp.qloguniform('learning_rate', np.log(.01),np.log(.1),.01),
    'subsample': hp.quniform('subsample', 0.5, 1.0,.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1.0,.1),
    'reg_alpha': hp.qloguniform('reg_alpha',np.log(1e-2),np.log(1e2),1e-2)
}
xgb_params = get_xgbparams(space,hyperopt_runs)
xgb_params