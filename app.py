from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and prepare your model here
df = pd.read_csv('C:/Users/91949/OneDrive/Desktop/Minor/training_data.csv')
X = df.iloc[:, :-1]
y = df['prognosis']
X_encoded = pd.get_dummies(X)
model = RandomForestClassifier(random_state=42)
model.fit(X_encoded, y)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data['symptoms']

    # Create input data structure
    input_data = pd.DataFrame({symptom: [0] for symptom in X_encoded.columns})
    for symptom in symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1
    
    # Make a prediction
    probabilities = model.predict_proba(input_data)[0]
    predicted_diseases = {disease: prob for disease, prob in zip(model.classes_, probabilities)}
    filtered_diseases = {disease: prob for disease, prob in predicted_diseases.items() if prob > 0.03}
    sorted_diseases = sorted(filtered_diseases.items(), key=lambda x: x[1], reverse=True)
    
    return jsonify({'predictions': sorted_diseases})

if __name__ == '__main__':
    app.run(debug=True)
