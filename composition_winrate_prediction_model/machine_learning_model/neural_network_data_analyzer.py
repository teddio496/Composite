import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import sys

from constants import STATS, CHALLENGE_STATS

# Result of the analysis





NUM_CHAMPIONS = 169
all_stats = {}

# Generate sequential values for STATS, CHALLEGE_STATS to analyze each 
# features separately, to see which has most impact on the game's result
for i, stat in enumerate(STATS):
    all_stats[stat] = [j for j in range(i + 1, i + 1 + 80, 8)]

for i, stat in enumerate(CHALLENGE_STATS, start=len(STATS)):
    all_stats[stat] = [j for j in range(i + 1, i + 1 + 80, 8)]



class LoLDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path, header=None)  
        data = data.iloc[:, 1:]                    
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


# Update the dataset class to filter only given list of features
class LoLDatasetTopFeatures(Dataset):
    def __init__(self, dataset, top_features):
        self.champ_ids = dataset.champ_ids  
        self.features = dataset.features[:, top_features]  # Filter only wanted features
        self.labels = dataset.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.cat([self.champ_ids[idx].flatten(), self.features[idx].flatten()]), self.labels[idx]


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



if __name__ == "__main__":

    for features in STATS + CHALLENGE_STATS:
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, "..", "data_preprocess", "data.csv")
        dataset = LoLDataset(data_path)

        top_features_dataset = LoLDatasetTopFeatures(dataset, all_stats[features])

        # Split dataset
        train_size = int(0.8 * len(top_features_dataset))
        test_size = len(top_features_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            top_features_dataset, [train_size, test_size]
        )

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = LoLModelOneHot(NUM_CHAMPIONS, len(all_stats[features])).to(device)

        # Training and evaluation
        def train_and_evaluate(model, train_dataloader, test_dataloader):
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

            for epoch in range(15):
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

            return correct / total

        # Train and evaluate
        accuracy = train_and_evaluate(model, train_dataloader, test_dataloader)
        print(f"Accuracy with {features}: {accuracy * 100:.2f}%")


