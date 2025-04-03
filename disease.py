from flask import Flask, render_template, request
import pickle
import numpy as np
import requests

app = Flask(__name__)

# Replace with your Google Form's action URL
FORM_ACTION_URL = "https://docs.google.com/forms/d/e/1FAIpQLScgD_f2ucw35BM80MWSin5Hp6RUIZFx9YOhSFLW-7hBAq-GuQ/formResponse"
FORM_FIELD_NAME = "entry.835636437"  # Replace with your field entry ID

# Load the model once at startup
model = pickle.load(open('models/diabetes.pkl', 'rb'))


def send_to_google_form(prediction):
    """Send the prediction to Google Form."""
    data = {FORM_FIELD_NAME: str(prediction)}
    requests.post(FORM_ACTION_URL, data=data)


def predict(symptoms):
    # Convert symptoms to a format suitable for the model (e.g., one-hot encoding)
    diabetes_symptoms = {'increased thirst', 'fatigue', 'blurred vision', 'dizziness', 'high blood sugar'}
    # Check if the entered symptoms match any combination of the diabetes symptoms
    if diabetes_symptoms.intersection(set(symptoms)):
        pred = 'Diabetes'
    else:
        pred = 'Unknown'
    send_to_google_form(pred)
    return pred


@app.route('/')
def home():
    return render_template('home2.html')


@app.route('/predict', methods=["POST"])
def predictPage():
    try:
        symptoms = request.form.values()
        symptoms = [symptom.lower() for symptom in symptoms]
        pred = predict(symptoms)
        return render_template('predict.html', pred=pred)
    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)
