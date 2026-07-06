from flask import Flask, request, render_template
import pickle
import numpy as np

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['AGE'])
        educ = int(request.form['EDUC'])
        married_input = request.form['MARRIED']
        married = 1 if married_input == "Married or With a Partner" else 2
        occat1_input = request.form['OCCAT1']
        occat1 = {
            "Manager Level": 1,
            "Mid-Level": 2,
            "Entry Level": 3,
            "Unemployed": 4
        }[occat1_input]
        kids = int(request.form['KIDS'])
        networth = float(request.form['NETWORTH'].replace(",", ""))
        income = float(request.form['INCOME'].replace(",", ""))
        yesfinrisk_input = request.form['YESFINRISK']
        yesfinrisk = 1 if yesfinrisk_input == "Yes" else 0
        debt = float(request.form['DEBT'].replace(",", ""))

        input_data = np.array([[age, educ, married, occat1, kids, networth, income, yesfinrisk, debt]])
        prediction = model.predict(input_data)

        return render_template('index.html', prediction_text=f'TrueRiskTolerance: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
