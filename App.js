import React, { useState } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [modelName, setModelName] = useState('MultinomialNB');

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  const handleModelChange = (e) => {
    setModelName(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, modelName }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('There has been a problem with your fetch operation:', error);
    }
  };

  return (
    <div className="app">
      <h1>Emotion Detection</h1>

      {/* Model Selection & Text Input */}
      <form onSubmit={handleSubmit} className="form-container">
        <label className="dropdown-label">Select Model:</label>
        <select value={modelName} onChange={handleModelChange} className="dropdown">
          {/*<option value="BERT">BERT (Accuracy: 93.62%)</option>*/}
          <option value="MultinomialNB">Multinomial Naive Bayes (Accuracy: 85.4%)</option>
          <option value="SVM">SVM (Accuracy: 89.48%)</option>
        </select>

        <textarea 
          value={text} 
          onChange={handleTextChange} 
          placeholder="Enter your text here..." 
          className="text-input"
        />

        <button type="submit" className="submit-btn">Submit</button>
      </form>

      {/* Model Description */}
      <div className="model-description">
        <h2> Model Information</h2>
        {modelName === "BERT" && (
          <p><strong>BERT (Bidirectional Encoder Representations from Transformers)</strong> is a deep learning-based NLP model that understands context using transformers. It offers state-of-the-art accuracy of <strong>93.62%</strong>.</p>
        )}
        {modelName === "MultinomialNB" && (
          <p><strong>Multinomial Naive Bayes</strong> is a probabilistic classifier based on Bayes' theorem, often used for text classification. It is lightweight and efficient, achieving an accuracy of <strong>85.4%</strong>.</p>
        )}
        {modelName === "SVM" && (
          <p><strong>Support Vector Machine (SVM)</strong> is a robust classifier that works well for text-based tasks. It finds the optimal hyperplane to separate classes and achieves an accuracy of <strong>89.48%</strong>.</p>
        )}
      </div>

      {/* Results Section */}
      {result && (
        <div className="result">
          <h2>🔍 Result</h2>
          <h3><strong>Predicted Emotion:</strong> {result.predictedEmotion}</h3>
          <h3>📊 Probabilities:</h3>
          <ul className="probability-list">
            {Object.keys(result.probabilities).map((emotion) => (
              <li key={emotion} className="probability-item">
                <strong>{emotion}</strong>: {result.probabilities[emotion]}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
