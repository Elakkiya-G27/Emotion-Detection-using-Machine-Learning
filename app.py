from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import os
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib  # Changed from pickle for SVM model loading

app = Flask(__name__)
CORS(app)

# Ensure device compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT Model & Tokenizer
model_path = "./saved_emotion_model"

if os.path.exists(model_path):
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        bert_model = BertForSequenceClassification.from_pretrained(model_path)
        bert_model.to(device)
        bert_model.eval()
        print("BERT Model Loaded Successfully")
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        bert_model, tokenizer = None, None
else:
    print("BERT model path not found. Ensure the model is saved in './saved_emotion_model'")
    bert_model, tokenizer = None, None

# Load Naïve Bayes Model & Vectorizer
try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        nb_vectorizer = pickle.load(f)
    
    with open("naive_bayes_emotion.pkl", "rb") as f:
        nb_model = pickle.load(f)
    
    if not hasattr(nb_vectorizer, "idf_"):
        raise ValueError("Naïve Bayes vectorizer is not fitted properly.")
    print("Naïve Bayes Model Loaded Successfully")
except Exception as e:
    print(f"Error loading Naïve Bayes model/vectorizer: {e}")
    nb_vectorizer, nb_model = None, None

# Load SVM Model & Vectorizer (using joblib)
try:
    svm_vectorizer = joblib.load("svm_tfidf_vectorizer.pkl")
    svm_model = joblib.load("svm_model.pkl")
    
    # Verify the vectorizer is fitted
    if not hasattr(svm_vectorizer, "idf_"):
        raise ValueError("SVM vectorizer is not fitted properly.")
    
    # Test prediction to verify model works
    test_input = svm_vectorizer.transform(["test input"])
    svm_model.predict(test_input)
    print("SVM Model and Vectorizer Loaded Successfully")
except Exception as e:
    print(f"Error loading SVM model/vectorizer: {e}")
    svm_vectorizer, svm_model = None, None

# Emotion labels
label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

def get_all_zero_probs():
    return {label: 0.0 for label in label_map.values()}

# Function to Predict with BERT
def predict_emotion_bert(text):
    if not bert_model or not tokenizer:
        return {"error": "BERT model not available"}, None
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        class_probs = probabilities.squeeze().cpu().numpy()
        
        result = {label_map[i]: round(float(class_probs[i]) * 100, 2) for i in range(len(class_probs))}
        predicted_class = label_map[class_probs.argmax()]
        return result, predicted_class
    except Exception as e:
        print(f"Error in BERT prediction: {e}")
        return {"error": str(e)}, None

# Function to Predict with Naïve Bayes
def predict_emotion_nb(text):
    if not nb_vectorizer or not nb_model:
        return {"error": "Naïve Bayes model not available"}, None
    try:
        text_tfidf = nb_vectorizer.transform([text])
        
        if hasattr(nb_model, 'predict_proba'):
            probabilities = nb_model.predict_proba(text_tfidf)[0]
            result = {label_map[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(probabilities))}
        else:
            prediction = nb_model.predict(text_tfidf)[0]
            result = get_all_zero_probs()
            result[label_map[prediction]] = 100.0
            
        predicted_class = max(result.items(), key=lambda x: x[1])[0]
        return result, predicted_class
    except Exception as e:
        print(f"Error in Naïve Bayes prediction: {e}")
        return {"error": str(e)}, None

# Function to Predict with SVM
def predict_emotion_svm(text):
    if not svm_vectorizer or not svm_model:
        return {"error": "SVM model not available"}, None
    try:
        text_tfidf = svm_vectorizer.transform([text])
        
        # Get probabilities (CalibratedClassifierCV should provide predict_proba)
        if hasattr(svm_model, 'predict_proba'):
            probabilities = svm_model.predict_proba(text_tfidf)[0]
            result = {label_map[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(probabilities))}
        else:
            # Fallback to decision function
            decision_values = svm_model.decision_function(text_tfidf)[0]
            exp_values = np.exp(decision_values - np.max(decision_values))
            probabilities = exp_values / exp_values.sum()
            result = {label_map[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(probabilities))}
        
        predicted_class = max(result.items(), key=lambda x: x[1])[0]
        return result, predicted_class
    except Exception as e:
        print(f"Error in SVM prediction: {e}")
        return {"error": str(e)}, None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "text" not in data or "modelName" not in data:
        return jsonify({'error': 'Invalid request format'}), 400
    
    text = data["text"].strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400
    
    model_name = data["modelName"]
    
    if model_name == 'BERT':
        probabilities, predicted_emotion = predict_emotion_bert(text)
    elif model_name == 'MultinomialNB':
        probabilities, predicted_emotion = predict_emotion_nb(text)
    elif model_name == 'SVM':
        probabilities, predicted_emotion = predict_emotion_svm(text)
    else:
        return jsonify({'error': 'Model not supported'}), 400

    if "error" in probabilities:
        return jsonify({'error': probabilities["error"]}), 500
    
    return jsonify({
        'probabilities': probabilities, 
        'predictedEmotion': predicted_emotion
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)