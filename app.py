from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import spacy
import numpy as np
import pandas as pd
from spacy.matcher import PhraseMatcher
from data_processing import load_data

app = Flask(__name__)
CORS(app)

# ✅ Load trained Random Forest model and MultiLabelBinarizer
rf_model = joblib.load("disease_model.pkl")
mlb = joblib.load("mlb_encoder.pkl")

# ✅ Load and preprocess data
df_symptoms, df_precaution = load_data()

# ✅ Load spaCy model
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# ✅ Prepare symptom patterns
all_symptoms = list(mlb.classes_)
matcher.add("SYMPTOMS", None, *[nlp(symptom) for symptom in all_symptoms])

# ✅ Function to extract symptoms from input text
def extract_symptoms(user_input):
    doc = nlp(user_input.lower())
    matches = matcher(doc)
    detected_symptoms = set(doc[start:end].text for _, start, end in matches)
    return list(detected_symptoms)

# ✅ API Route to accept user symptoms and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        user_input = data.get("symptoms", "")

        detected_symptoms = extract_symptoms(user_input)
        input_vector = [1 if symptom in detected_symptoms else 0 for symptom in mlb.classes_]

        predicted_disease = rf_model.predict([input_vector])[0]
        precautions = df_precaution[df_precaution["Disease"] == predicted_disease].iloc[:, 1:].dropna(axis=1).values.flatten()

        return jsonify({
            "predicted_disease": predicted_disease,
            "precautions": list(precautions) if precautions.size > 0 else ["No precautions found."]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
