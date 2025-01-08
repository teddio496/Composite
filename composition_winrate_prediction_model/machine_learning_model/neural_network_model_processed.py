import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class LoLDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path, header=None)

        # Map labels (first column) to binary and handle missing/infinite values
        data.iloc[:, 0] = data.iloc[:, 0].replace([np.inf, -np.inf], np.nan).fillna(0)
        data.iloc[:, 0] = data.iloc[:, 0].map({100: 1, 200: 0}).astype(int)

        # Extract labels
        self.labels = data.iloc[:, 0].values

        # Identify champion ID columns (integer columns)
        champ_id_indices = list(range(1, 89, 9))  
        print(champ_id_indices)
        self.champ_ids = data.iloc[:, champ_id_indices].values

        # Extract numerical feature columns
        feature_indices = list(range(11, data.shape[1]))
        self.features = data.iloc[:, feature_indices].values

        # Handle missing/infinite values in features
        self.features = pd.DataFrame(self.features).replace([np.inf, -np.inf], np.nan).fillna(0).values

        # Normalize numerical features
        scaler = MinMaxScaler()
        self.features = scaler.fit_transform(self.features)

        # Convert to PyTorch tensors
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.champ_ids = torch.tensor(self.champ_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.cat([self.champ_ids[idx].flatten(), self.features[idx].flatten()]), self.labels[idx]


# Define Neural Network Model
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
        champ_ids = x[:, :10].long()  # First 10 columns are champion IDs
        features = x[:, 10:]         # Remaining columns are features

        # Clamp champion IDs to valid range
        champ_ids = champ_ids.clamp(0, NUM_CHAMPIONS - 1)
        batch_size = champ_ids.size(0)

        # One-hot encode champion IDs
        one_hot_champions = torch.zeros(batch_size, 10, NUM_CHAMPIONS, device=x.device)
        one_hot_champions.scatter_(2, champ_ids.unsqueeze(2), 1)
        one_hot_champions = one_hot_champions.view(batch_size, -1)

        # Process numerical features
        processed_features = self.feature_layer(features)

        # Combine one-hot encoded IDs and processed features
        combined = torch.cat([one_hot_champions, processed_features], dim=1)

        return self.output_layer(combined)


# Main Training and Evaluation
if __name__ == "__main__":
    # Dataset path
    data_path = "data.csv"  # Update this path if necessary
    dataset = LoLDataset(data_path)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Training data size: {len(train_dataset)}")
    print(f"Testing data size: {len(test_dataset)}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CHAMPIONS = 10000  # Set based on the maximum champion ID in your dataset
    NUM_FEATURES = dataset.features.shape[1]  # Number of numerical features

    # Initialize model
    model = LoLModelOneHot(NUM_CHAMPIONS, NUM_FEATURES).to(device)

    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Training loop
    for epoch in range(15):
        model.train()
        total_loss = 0
        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
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


# Save the trained model
model_path = "lol_model.pkl"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_champions': NUM_CHAMPIONS,
    'num_features': NUM_FEATURES,
}, model_path)
print(f"Model saved to {model_path}")
