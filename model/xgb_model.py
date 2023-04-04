import pickle
import os
file_path = os.path.abspath('../mlb_data/dataframe.pkl')
df = pickle.load(open(file_path, 'rb'))

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
X_test = X[-1002:]
y_test = y[-1002:]

import xgboost as xgb
params = {'learning_rate': 0.03 ,'max_depth': 3, 'colsample_bytree': .6, 'min_child_weight': 2.0, 'subsample': 0.600000000000001 , 'reg_alpha': 6.08, 'tree_method': 'gpu_hist'}



#params = {'colsample_bytree': 0.5, 'learning_rate': 0.08, 'max_depth': 5, 'min_child_weight': 4.0, 'reg_alpha': 2.8000000000000003, 'subsample': 1.0, 'gpu_id': 0, 'tree_method': 'gpu_hist', 'seed': 13}
gbm = xgb.XGBClassifier(**params)
model = gbm.fit(X_train, y_train,
                eval_set = [[X_train, y_train],
                          [X_valid, y_valid]],
                eval_metric='logloss',
                early_stopping_rounds=10)

xgb_test_preds = model.predict(X_test)
xgb_test_proba = model.predict_proba(X_test)[:,1]

from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss
import matplotlib.pyplot as plt
import pickle

def cal_curve(data, bins):
    # adapted from:
    #https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    fig = plt.figure(1, figsize=(12, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for y_test, y_pred, y_proba, name in data:
        brier = brier_score_loss(y_test, y_proba)
        print("{}\t\tAccuracy:{:.4f}\t Brier Loss: {:.4f}".format(
            name, accuracy_score(y_test, y_pred), brier))
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, y_proba, n_bins=bins)
        ax1.plot(mean_predicted_value, fraction_of_positives,
                 label="%s (%1.4f)" % (name, brier))
        ax2.hist(y_proba, range=(0, 1), bins=bins, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
import os
file_path = os.path.abspath('../covers_data/baseline.pkl')
outcomes,predictions,probabilities= pickle.load(open(file_path, 'rb'))
 
data = [
    (outcomes, predictions, probabilities, 'Casino'),
    (y_test,xgb_test_preds, xgb_test_proba, 'XGBoost')
]
cal_curve(data, 15)
import pandas as pd
x = pd.Series(model.get_booster().get_score(importance_type= 'total_gain')
         ).sort_values()
_ = x[-25:].plot(kind='barh',title="XGBoost Feature Gain")
from matplotlib import pyplot
results = model.evals_result()
pyplot.plot(results['validation_0']['logloss'], label='train')
pyplot.plot(results['validation_1']['logloss'], label='test')
pyplot.legend()
pyplot.show()

import pickle
pickle.dump(model,open('xgb_model.pkl','wb'))



print(X_valid.shape)
print(y_valid.shape)
print(df.shape)
from sklearn.metrics import roc_auc_score

# assuming you have already trained your model and generated predictions
proba = model.predict_proba(X_test)[:, 1] # assuming you want the AUC for the positive class
auc = roc_auc_score(y_test, proba)
print('AUC:', auc)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print(X_test.shape)

