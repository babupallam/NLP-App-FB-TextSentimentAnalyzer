# models/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from model import SentimentAnalysisModel
from utils import preprocess_data, create_dataloaders

def train_model(data_file, input_dim, embedding_dim, hidden_dim, output_dim, epochs=10):
    data = preprocess_data(data_file)
    train_loader, test_loader = create_dataloaders(data)
    model = SentimentAnalysisModel(input_dim, embedding_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for text, label in train_loader:
            optimizer.zero_grad()
            predictions = model(text)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss}')

    torch.save(model.state_dict(), 'saved_models/sentiment_model.pth')

if __name__ == "__main__":
    train_model('data/sentiment_data.csv', input_dim=5000, embedding_dim=100, hidden_dim=256, output_dim=3)
