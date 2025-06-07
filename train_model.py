import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the processed data
df = pd.read_csv("processed_data.csv")

# Load the MultiLabelBinarizer
mlb = joblib.load("mlb_encoder.pkl")

# Extract features and target
X = df.drop(columns=["Disease"]).values  # Use one-hot encoded symptom columns
y = df["Disease"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "disease_model.pkl")

print("✅ Model trained and saved successfully!")

# Function to predict disease
def predict_disease(user_symptoms):
    """Predict disease based on user input symptoms."""
    user_symptoms = [s.strip().lower() for s in user_symptoms]  # Normalize input
    user_vector = [1 if symptom in user_symptoms else 0 for symptom in mlb.classes_]  # Create binary input vector

    prediction = model.predict([user_vector])
    return prediction[0]  # Return predicted disease
