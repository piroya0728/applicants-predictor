from flask import Flask, render_template, request
import pandas as pd
import csv

import pickle



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        f = request.form['csvfile']
        l = []
        with open(f, encoding='utf-8') as file:
            csvfile= csv.reader(file)
            for row in csvfile:
                l.append(row)
        X_df = pd.DataFrame(l[1:], columns=l[0])
            

        X = X_df[['正社員登用あり', 'フラグオプション選択', '給与/交通費　給与下限', '勤務地　都道府県コード', 'お仕事名']]
        
        X['正社員登用あり'] =X['正社員登用あり'].astype(float)
        X['給与/交通費　給与下限'] =X['給与/交通費　給与下限'].astype(float)


        #「フラグオプション選択」の特徴量作成
        X_flag = []
        for i in range(X.shape[0]):
            if X['フラグオプション選択'][i] == 3 or X['フラグオプション選択'][i] == 5:
                X_flag.append(1)
            else:
                X_flag.append(0)
        X = X.drop(['フラグオプション選択'], axis=1)
        X['FLAG'] = X_flag


        #東京or神奈川  vs  地方  特徴量作成
        tokyo_or_kanagawa = []
        for i in range(X.shape[0]):
            if X['勤務地　都道府県コード'][i] == 13 or X['勤務地　都道府県コード'][i] == 14:
                tokyo_or_kanagawa.append(1)
            else:
                tokyo_or_kanagawa.append(0)
        X = X.drop(['勤務地　都道府県コード'], axis=1)
        X['tokyo_or_kanagawa'] = tokyo_or_kanagawa
        

        #賞与に関する特徴量
        bonus = []
        for i in range(X.shape[0]):
            if '賞与' in X['お仕事名'][i] or 'ボーナス' in X['お仕事名'][i]:
                bonus.append(1)
            else:
                bonus.append(0)
        X = X.drop(['お仕事名'], axis=1)
        X['bonus'] = bonus
    



        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X)
        X_df['応募数 合計'] = y_pred
        predicted_file = X_df[['お仕事No.', '応募数 合計']]

        predicted_file.to_csv('static/predicted_file.csv', index=False)

        return  render_template('data.html')   


if __name__ == '__main__':
    app.run(debug=False)