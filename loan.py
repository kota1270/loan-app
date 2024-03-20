from flask import Flask, request, render_template, jsonify
#import pandas as pd
import numpy as np
import joblib
import io, csv
import os

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/', methods=['GET'])
def home():
    return render_template('loan_default_prediction.html')  # ホームページのテンプレートを表示

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        filebuf = request.files.get('csvfile')
        print(request.files)
        if filebuf is None:
            return jsonify(message='ファイルを指定してください'), 400
        elif 'text/csv' != filebuf.mimetype:
            return jsonify(message='CSVファイル以外は受け付けません'), 415

        text_stream = io.TextIOWrapper(filebuf.stream, encoding='cp932')
        data = []
        for row in csv.reader(text_stream):
            data.append(row)
        prediction = model.predict(data)
        return render_template('loan_default_prediction.html', prediction=prediction)

    return render_template('loan_default_prediction.html')  # 特徴量の入力フォームを表示

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)






