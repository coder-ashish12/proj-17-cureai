import joblib
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from data_processing import load_data
from train_model import predict_disease

# ✅ Load trained Random Forest model and MultiLabelBinarizer
rf_model = joblib.load("disease_model.pkl")
mlb = joblib.load("mlb_encoder.pkl")

# ✅ Load and preprocess data
df_symptoms, df_precaution = load_data()

# ✅ Load spaCy model for NLP-based symptom extraction
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# ✅ Prepare symptom patterns for NLP extraction
all_symptoms = list(mlb.classes_)
symptom_patterns = [nlp(symptom) for symptom in all_symptoms]
matcher.add("SYMPTOMS", None, *symptom_patterns)

# ✅ Function to extract symptoms using NLP
def extract_symptoms(user_input):
    doc = nlp(user_input.lower())
    matches = matcher(doc)
    detected_symptoms = set(doc[start:end].text for _, start, end in matches)
    return list(detected_symptoms)

# ✅ Get user input in natural language
user_input = input("Enter your symptoms prompt:\n")

# ✅ Extract symptoms using NLP
detected_symptoms = extract_symptoms(user_input)
print("🔹 Detected Symptoms:", detected_symptoms)

# ✅ Convert detected symptoms into binary input for Random Forest
input_vector = [1 if symptom in detected_symptoms else 0 for symptom in mlb.classes_]

# ✅ Predict disease using trained model
predicted_disease = rf_model.predict([input_vector])[0]
print("🩺 Predicted Disease:", predicted_disease)

# ✅ Retrieve precautions
precautions = df_precaution[df_precaution["Disease"] == predicted_disease].iloc[:, 1:].dropna(axis=1).values.flatten()
if precautions.size > 0:
    print("🛑 Recommended Precautions:")
    for i, precaution in enumerate(precautions, 1):
        print(f"   {i}. {precaution}")
else:
    print("⚠️ No precautions found for this disease.")