from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
import pickle


app = Flask(__name__)

model = load_model('insurance_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/insurance_model')
def  insurance():
    return render_template('form.html') 




@app.route('/submit', methods=['POST'])
def submit():
    age = request.form['age']
    gender = request.form['gender']
    bmi = request.form['bmi']
    children = request.form['children']
    smoker = request.form['smoker']
    region = request.form['region']
    encoder_region = pickle.load(open('insurance_encoder_region.pkl','rb'))
    # print("Before opening file")
    # try:
    #  with open('insurance_encoder_smoker.pkl', 'rb') as f:
    #     encoder_region = pickle.load(f)
    #     print("File opened successfully")
    # except FileNotFoundError:
    #  print("The file 'insurance_encoder_smoker.pkl' was not found.")
    # except Exception as e:
    #  print("An error occurred while loading the pickled file:", e)
    region_code = encoder_region.transform([region])[0]
    # if region=="northeast":
    #     region_code=0
    # elif region=="northwest":
    #     region_code=1
    # elif region=="southeast":
    #     region_code=2
    # else:
    #     region_code=3


    if gender=="male" :
        gender_code=1
    else:
        gender_code=0  

    if smoker=="yes" :
        smoker_code=1
    else:
        smoker_code=0      

    data = {
            'age': [age],
            'sex': [gender_code],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker_code],
            'region': [region_code]
        }   
    df = pd.DataFrame(data)
    with open('insurance_data_scaler.pkl', 'rb') as f:
       scaler = pickle.load(f)
    X = pd.DataFrame(scaler.transform(df), columns=df.columns)
    y=model.predict(X)
    y=float(y)
    return f"Insurance : {y}"


    # return f"age: {age}, Gender: {gender_code}, bmi: {bmi}, Smoker: {smoker_code} ,Children: {children} , Region: {region_code}"
    # return X['age'],X['sex'],X['bmi' ],X['children'] ,X['smoker'],X['region']
    # return jsonify(X.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(port=8000,debug="true")