import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np


class LoLDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path, header=None)  
        data = data.iloc[:, 1:]                     # Drop match ID

        data.iloc[:, 0] = data.iloc[:, 0].map({100: 1, 200: 0})

        self.labels = data.iloc[:, 0].values            # Team 1 win/loss
        self.champ_ids = data.iloc[:, 1:11].values      # Champion IDs
        self.features = data.iloc[:, 11:].values        # Numerical features

        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        # Normalize numerical features
        scaler = MinMaxScaler()
        self.features = scaler.fit_transform(self.features)
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.champ_ids = torch.tensor(self.champ_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.cat([self.champ_ids[idx].flatten(), self.features[idx].flatten()]), self.labels[idx]


class LoLChampionDataset(Dataset):
    # Only grab the champion IDs, no extra features
    def __init__(self, file_path, num_champions=169):  # Default number of champions
        data = pd.read_csv(file_path, header=None)  
        data = data.iloc[:, 1:]                     
        data.iloc[:, 0] = data.iloc[:, 0].map({100: 1, 200: 0})

        # Clip champion IDs to valid range
        self.champ_ids = data.iloc[:, 1:11].values
        self.champ_ids = np.clip(self.champ_ids, 0, num_champions - 1)
        self.champ_ids = torch.tensor(self.champ_ids, dtype=torch.long)

        # Convert labels to torch tensor
        self.labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.champ_ids[idx], self.labels[idx]


class LoLModelOneHot(nn.Module):
    def __init__(self, num_champions, num_features):
        super(LoLModelOneHot, self).__init__()
        
        # Linear layers for numerical features
        self.feature_layer = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Final classification layer
        self.output_layer = nn.Sequential(
            nn.Linear(10 * num_champions + 32, 64),  
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        champ_ids = x[:, :10].long() 
        features = x[:, 10:] 

        champ_ids = champ_ids.clamp(0, NUM_CHAMPIONS - 1)

        batch_size = champ_ids.size(0)
        one_hot_champions = torch.zeros(batch_size, 10, NUM_CHAMPIONS, device=x.device)
        one_hot_champions.scatter_(2, champ_ids.unsqueeze(2), 1)

        one_hot_champions = one_hot_champions.view(batch_size, -1)

        processed_features = self.feature_layer(features)

        combined = torch.cat([one_hot_champions, processed_features], dim=1)

        return self.output_layer(combined)


class LoLChampionModelOneHot(nn.Module):
    def __init__(self, num_champions):
        super(LoLChampionModelOneHot, self).__init__()
        
        # Classification layers
        self.classification_layer = nn.Sequential(
            nn.Linear(10 * num_champions, 64),  # Input: concatenated one-hot encodings
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        champ_ids = x[:, :10].long()  # Champion IDs
        champ_ids = champ_ids.clamp(0, NUM_CHAMPIONS - 1)

        # One-hot encoding
        batch_size = champ_ids.size(0)
        one_hot_champions = torch.zeros(batch_size, 10, NUM_CHAMPIONS, device=x.device)
        one_hot_champions.scatter_(2, champ_ids.unsqueeze(2), 1)

        # Flatten one-hot encodings
        one_hot_champions = one_hot_champions.view(batch_size, -1)

        # Classification
        return self.classification_layer(one_hot_champions)


class LoLChampionModelEmbedding(nn.Module):
    def __init__(self, num_champions, embedding_dim):
        super(LoLChampionModelEmbedding, self).__init__()

        # Embedding layer for champion IDs
        self.embedding = nn.Embedding(num_champions, embedding_dim)

        # Classification layers
        self.classification_layer = nn.Sequential(
            nn.Linear(10 * embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout to prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        champ_ids = x[:, :10].long()  # Champion IDs
        batch_size = champ_ids.size(0)

        # Embedding champions
        champ_embeds = self.embedding(champ_ids).view(batch_size, -1)  # Flatten embeddings

        # Classification
        return self.classification_layer(champ_embeds)      



if __name__ == "__main__":
    # Load dataset
    current_dir = os.path.dirname(__file__)
    # data_path = os.path.join(current_dir, "..", "data-preprocess", "data.csv")
    data_path = "data_test.csv"
    dataset = LoLDataset(data_path)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Training data size: {len(train_dataset)}")
    print(f"Testing data size: {len(test_dataset)}")

    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    NUM_CHAMPIONS = 169  # Adjust based on the number of champions
    NUM_FEATURES = 81   # Dimension of embedding vectors

    # Initialize the model
    model = LoLModelOneHot(NUM_CHAMPIONS, NUM_FEATURES).to(device)

    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # L2 regularization

    # Training loop
    for epoch in range(5):
        model.train()
        total_loss = 0
        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")