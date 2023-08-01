from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import datetime
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Assume you have saved the scaler object to disk

@app.route('/')
def home():
    current_year = datetime.now().year
    return render_template('index.html', year=current_year)
@app.route('/predict', methods=['POST'])
def predict():
    Length1 = float(request.form.get('length1'))
    Length2 = float(request.form.get('length2'))
    Length3 = float(request.form.get('length3'))
    Height = float(request.form.get('height'))
    Width = float(request.form.get('width'))

    # Build a feature array and reshape it for prediction
    features = np.array([Length1, Length2, Length3, Height, Width]).reshape(1, -1)

    # Scale the features
    features = scaler.transform(features)

    prediction = model.predict(features)

    return render_template('index.html', prediction_text='Predicted Species is {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)

