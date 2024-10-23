# models/model.py
import torch
import torch.nn as nn

class SentimentAnalysisModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(SentimentAnalysisModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.rnn(embedded)
        output = self.fc(hidden[-1])
        return self.softmax(output)
