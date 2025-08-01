from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        pclass = int(request.form['pclass'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])

        # Prepare and scale data
        input_data = np.array([[age, fare, pclass, sibsp, parch]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)

        result = 'Survived' if prediction[0] == 1 else 'Did Not Survive'
        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
