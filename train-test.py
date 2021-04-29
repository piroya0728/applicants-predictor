import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import lightgbm as lgb

X_df = pd.read_csv("./train_x.csv")
y_df = pd.read_csv("./train_y.csv")
X = X_df[['正社員登用あり','フラグオプション選択','給与/交通費　給与下限','勤務地　都道府県コード', 'お仕事名']]
y = y_df['応募数 合計']

#新しい特徴量:X_flag
# 'フラグオプション選択'のカラムが３か５の時に1、それ以外は0となるダミー変数
X_flag = []
for i in range(X.shape[0]):
    if X['フラグオプション選択'][i] == 3 or X['フラグオプション選択'][i] == 5:
        X_flag.append(1)
    else:
        X_flag.append(0)
X = X.drop(['フラグオプション選択'], axis=1)
X['FLAG'] = X_flag

tokyo_or_kanagawa = []
for i in range(X.shape[0]):
    if X['勤務地　都道府県コード'][i] == 13 or X['勤務地　都道府県コード'][i] == 14:
        tokyo_or_kanagawa.append(1)
    else:
        tokyo_or_kanagawa.append(0)
X = X.drop(['勤務地　都道府県コード'], axis=1)
X['tokyo_or_kanagawa'] = tokyo_or_kanagawa

bonus = []
for i in range(X.shape[0]):
    if '賞与' in X['お仕事名'][i] or 'ボーナス' in X['お仕事名'][i]:
        bonus.append(1)
    else:
        bonus.append(0)
X = X.drop(['お仕事名'], axis=1)
X['bonus'] = bonus


y = np.array(y)



X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.20, shuffle=True, random_state=0)
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)




with open('model.pickle', mode='wb') as f:
    pickle.dump(model, f)