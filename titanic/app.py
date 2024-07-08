from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

with open("data.json", 'r') as f:
    validation = pd.read_json(f)
all_data = validation.to_dict('records')

with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

@app.route('/', methods=['POST', 'GET'])
def hello():
    return render_template("home.html", all_data=all_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict(flat=False)
        input_df = pd.DataFrame.from_dict(data, orient='index').T
        input_encoded = input_df.copy()
        for column in input_df.columns:
            input_encoded[column] = encoder.fit_transform(input_df[column])
            
        encoded_data = input_encoded.values[0]
        prediction = model.predict([encoded_data])[0]
        
        return render_template("home.html", all_data=all_data, prediction=prediction)
    except Exception as e:
        error = str(e)
        return render_template("home.html", all_data=all_data, error=error)

if __name__ == "__main__":
    app.run(debug=True)
