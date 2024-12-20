import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import h5py
import os
import time

DATA_DIR = "datasets/Galaxy10_DECals.h5"
NUM_SAMPLES = 500

# Define a simple CNN for regression
class RedshiftCNN(nn.Module):
    def __init__(self):
        super(RedshiftCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 36 * 36, 128),  # Assuming input size is (144, 144, 3)
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for regression
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x

# Custom dataset class
class GalaxyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(Image.fromarray(image.astype('uint8')).convert("RGB"))
        return image, torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    # Load dataset
    with h5py.File(DATA_DIR, 'r') as f:
        images = np.array(f['images'][:NUM_SAMPLES])
        labels = np.array(f['redshift'][:NUM_SAMPLES])

        # Filter to data where redshift is known
        valid_indices = ~np.isnan(labels)
        images = images[valid_indices]
        labels = labels[valid_indices]

        print(f"{labels.shape[0]} valid samples loaded")

        # Split the data into training and testing sets
        images_train, images_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
        )

    mean = np.mean(images, axis=(0, 1, 2)) / 255.0  # Normalize by 255
    std = np.std(images, axis=(0, 1, 2)) / 255.0

    # Preprocessing and transforms
    transform = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # Create datasets for training and testing
    train_dataset = GalaxyDataset(images_train, labels_train, transform=transform)
    test_dataset = GalaxyDataset(images_test, labels_test, transform=transform)

    # Create dataloaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss, and optimizer
    model = RedshiftCNN()
    model.to(torch.device('cpu'))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    start = time.time()
    epochs = 30
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(torch.device('cpu')), targets.to(torch.device('cpu'))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    end = time.time()
    print(f"Training took {end - start} seconds")
    # Evaluation on the test set
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(torch.device('cpu'))
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().tolist())
            true_labels.extend(targets.tolist())

    r2 = r2_score(true_labels, predictions)
    print(f"R-squared score on test set: {r2:.4f}")
