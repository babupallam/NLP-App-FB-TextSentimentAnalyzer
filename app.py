# app.py
from flask import Flask, render_template, request, jsonify
import torch
from models.model import SentimentAnalysisModel
from models.utils import preprocess_data

app = Flask(__name__)

model = SentimentAnalysisModel(input_dim=5000, embedding_dim=100, hidden_dim=256, output_dim=3)
model.load_state_dict(torch.load('saved_models/sentiment_model.pth'))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    tokenized = preprocess_data(text)  # Assume this function handles single input as well
    with torch.no_grad():
        prediction = model(tokenized)
    sentiment = torch.argmax(prediction).item()
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
