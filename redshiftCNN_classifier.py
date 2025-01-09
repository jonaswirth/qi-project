# Correcting the CNN fully connected layer dimensions and fixing training logic
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import h5py
import os
import time
import matplotlib.pyplot as plt

RANDOM_SEED = 42
NUM_CLASSES = 10

mean_bins = []

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_DIR = "datasets/Galaxy10_DECals.h5"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define a simple CNN for classification
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

        # Placeholder for dynamically calculating the input size
        self._flattened_size = None

        # The fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(1, 1),  # Placeholder; will be updated later
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)

        # Update the first Linear layer dynamically if not already set
        if self._flattened_size is None:
            self._flattened_size = x.size(1)
            self.fc_layers[0] = nn.Linear(self._flattened_size, 128)

        x = self.fc_layers(x)
        return x


# Custom dataset class
class GalaxyDataset(Dataset):
    def __init__(self, images, labels, class_labels, transform=None):
        self.images = images
        self.labels = labels
        self.class_labels = class_labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.class_labels[idx]
        if self.transform:
            image = self.transform(Image.fromarray(image.astype('uint8')).convert("RGB"))
        return image, torch.tensor(label, dtype=torch.long)  # Ensure labels are long type

from sklearn.metrics import r2_score

if __name__ == "__main__":
    def train_and_evaluate(class_index):
        # Load dataset
        with h5py.File(DATA_DIR, 'r') as f:
            images = np.array(f['images'])
            classes = np.array(f['ans'])
            labels = np.array(f['redshift'])

            class_indices = np.where(classes == class_index)
            images = images[class_indices]
            labels = labels[class_indices]

            valid_indices = ~np.isnan(labels)
            images = images[valid_indices]
            labels = labels[valid_indices]

            # Map labels to class indices
            global mean_bins
            bins = np.linspace(np.min(labels), np.max(labels), NUM_CLASSES + 1)
            class_labels = np.digitize(labels, bins, right=False) - 1
            class_labels = np.clip(class_labels, 0, NUM_CLASSES - 1)

            # Compute the mean value for each bin
            mean_bins = []
            for i in range(NUM_CLASSES):
                bin_values = labels[class_labels == i]
                if len(bin_values) > 0:
                    mean_bins.append(bin_values.mean())
                else:
                    mean_bins.append(0)  # Or np.nan, depending on your needs

            mean_bins = np.array(mean_bins)

            print(f"{labels.shape[0]} valid samples loaded")
            print(images.shape, labels.shape)

            images_train, images_temp, labels_train, labels_temp, class_labels_train, class_labels_temp = train_test_split(
                images, labels, class_labels, test_size=0.3, random_state=RANDOM_SEED
            )
            images_val, images_test, labels_val, labels_test, class_labels_val, class_labels_test = train_test_split(
                images_temp, labels_temp, class_labels_temp, test_size=0.5, random_state=RANDOM_SEED
            )

        mean = np.mean(images, axis=(0, 1, 2)) / 255.0
        std = np.std(images, axis=(0, 1, 2)) / 255.0

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        ])

        train_dataset = GalaxyDataset(images_train, labels_train, class_labels_train, transform=transform)
        val_dataset = GalaxyDataset(images_val, labels_val, class_labels_val, transform=transform)
        test_dataset = GalaxyDataset(images_test, labels_test, class_labels_test, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = RedshiftCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        epochs = 15
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
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
                    val_loss += criterion(outputs, targets).item() * inputs.size(0)

            val_loss /= len(val_dataset)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        actual_values = []
        predicted_values = []
        reconstructed_values = []
        shown_results = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Convert outputs to probabilities
                probabilities = torch.softmax(outputs, dim=1)

                # Predict class labels
                _, predicted = torch.max(probabilities, 1)

                # Reconstruct continuous values
                reconstructed = probabilities @ torch.tensor(mean_bins, dtype=torch.float32, device=device)

                # Accumulate results for evaluation
                actual_values.extend(targets.cpu().numpy())
                predicted_values.extend(predicted.cpu().numpy())
                reconstructed_values.extend(reconstructed.cpu().numpy())

                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                if shown_results < 5:
                    print(f"Redshift: {targets.cpu().numpy()} \nReconstructed: {reconstructed.cpu().numpy()} \nProbabilities: {probabilities}\nPredicted: {predicted.cpu().numpy()}")
                    shown_results += 1

        print(mean_bins)
        # Compute accuracy
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

        # Compute R² score for reconstructed values
        r2 = r2_score(actual_values, reconstructed_values)
        print(f"R² Score: {r2:.4f}")

        return accuracy, r2

    stats = []
    for n in range(0, 1):  # Change range for other classes
        accuracy, r2 = train_and_evaluate(n)
        stats.append([n, accuracy, r2])

    stats = np.array(stats)

    plt.plot(stats[:, 0], stats[:, 1])
    plt.title("Test Accuracy")
    plt.savefig("images/redshiftCNN/test_accuracy.png")
    plt.clf()

    plt.plot(stats[:, 0], stats[:, 2])
    plt.title("R² Score")
    plt.savefig("images/redshiftCNN/r2_score.png")

