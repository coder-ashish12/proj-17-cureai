import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def load_data():
    """Loads and preprocesses the disease and symptom dataset."""
    symptoms_df = pd.read_csv("DiseaseAndSymptoms.csv")
    precautions_df = pd.read_csv("Disease precaution.csv")  # Fixed file name typo

    # Fill NaN values with empty strings
    symptoms_df.fillna("", inplace=True)

    # Split symptoms into lists
    symptoms_df["Symptoms_List"] = symptoms_df.iloc[:, 1:].apply(lambda x: [s.strip() for s in x.dropna()], axis=1)

    # One-hot encode symptoms using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    symptoms_encoded = mlb.fit_transform(symptoms_df["Symptoms_List"])
    
    # Create a DataFrame with the one-hot encoded symptoms
    symptoms_encoded_df = pd.DataFrame(symptoms_encoded, columns=mlb.classes_)
    
    # Add the disease column back
    processed_df = pd.concat([symptoms_df[["Disease"]], symptoms_encoded_df], axis=1)

    # ✅ Save processed data
    processed_df.to_csv("processed_data.csv", index=False)
    joblib.dump(mlb, "mlb_encoder.pkl")  # Save MultiLabelBinarizer

    print("✅ Processed Data and Encoder Saved Successfully")

    return processed_df, precautions_df  # ✅ Returning only two values

if __name__ == "__main__":
    load_data()
