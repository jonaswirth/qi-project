### Simple classical CNN for estimating redshift from image data

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
import matplotlib.pyplot as plt

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_DIR = "datasets/Galaxy10_DECals.h5"

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define a simple CNN for regression
class RedshiftCNN(nn.Module):
    def __init__(self):
        super(RedshiftCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),  # Assuming input size is (144, 144, 3)
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1)  # Single output for regression
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
    def train_and_evaluate(class_index):
        # Load dataset
        with h5py.File(DATA_DIR, 'r') as f:
            images = np.array(f['images'])
            classes = np.array(f['ans'])
            labels = np.array(f['redshift'])

            # Filter to one type of galaxy
            class_indeces = np.where(classes == class_index)
            images = images[class_indeces]
            labels = labels[class_indeces]
             
            # Filter to data where redshift is known
            valid_indices = ~np.isnan(labels)
            images = images[valid_indices]
            labels = labels[valid_indices]

            print(f"{labels.shape[0]} valid samples loaded")
            print(images.shape)
            print(labels.shape)

            # Split the data into training, validation, and testing sets
            images_train, images_temp, labels_train, labels_temp = train_test_split(
                images, labels, test_size=0.3, random_state=RANDOM_SEED
            )
            images_val, images_test, labels_val, labels_test = train_test_split(
                images_temp, labels_temp, test_size=0.5, random_state=RANDOM_SEED
            )

            print(labels_train.shape, labels_val.shape, labels_test.shape)

        mean = np.mean(images, axis=(0, 1, 2)) / 255.0  # Normalize by 255
        std = np.std(images, axis=(0, 1, 2)) / 255.0

        # Preprocessing and transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        ])

        # Create datasets for training, validation, and testing
        train_dataset = GalaxyDataset(images_train, labels_train, transform=transform)
        val_dataset = GalaxyDataset(images_val, labels_val, transform=transform)
        test_dataset = GalaxyDataset(images_test, labels_test, transform=transform)

        # Create dataloaders for training, validation, and testing
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model, loss, and optimizer
        model = RedshiftCNN()
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        start = time.time()
        epochs = 30
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.view(-1)
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataset)

            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs.view(-1), targets.view(-1)).item() * inputs.size(0)

            val_loss = val_loss / len(val_dataset)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")

        end = time.time()
        training_time = end - start
        print(f"Training took {training_time} seconds")

        # Evaluation on the test set
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)       
                outputs = outputs.view(-1)
                predictions.extend(outputs.tolist())
                true_labels.extend(targets.view(-1).tolist())

        r2 = r2_score(true_labels, predictions)
        print(f"R-squared score on test set: {r2:.4f}")

        return training_time, r2

    # Run for each class of galaxy individually
    stats = []

    for n in range(0, 1):
        training_time, r2 = train_and_evaluate(n)
        stats.append([n, training_time, r2])
    
    stats = np.array(stats)

    plt.plot(stats[:,0], stats[:,1])
    plt.title("Training time")
    plt.savefig(f"images/redshiftCNN/training_time.png")
    plt.clf()

    plt.plot(stats[:,0], stats[:,2])
    plt.title("R-squared score")
    plt.savefig(f"images/redshiftCNN/scores.png")
    plt.clf()
