# models/utils.py
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['label'] = LabelEncoder().fit_transform(df['sentiment'])
    tokenizer = get_tokenizer('basic_english')
    df['tokenized'] = df['text'].apply(lambda x: tokenizer(x))
    return df

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = torch.tensor(self.data.iloc[idx]['tokenized'])
        label = torch.tensor(self.data.iloc[idx]['label'])
        return text, label

def create_dataloaders(data, batch_size=32):
    train_data, test_data = train_test_split(data, test_size=0.2)
    train_dataset = SentimentDataset(train_data)
    test_dataset = SentimentDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader
