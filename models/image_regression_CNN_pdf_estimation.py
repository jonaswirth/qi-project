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

DATA_DIR = "../datasets/astroclip_reduced_2.h5"

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Binning Function
def create_bins(labels, n_bins):
    min_redshift, max_redshift = labels.min(), labels.max()
    bins = np.linspace(min_redshift, max_redshift, n_bins + 1)
    bin_indices = np.clip(np.digitize(labels, bins) - 1, 0, n_bins - 1)
    bin_means = (bins[:-1] + bins[1:]) / 2  # Bin centers
    return bins, bin_indices, bin_means

def create_adaptive_bins(labels, n_bins):
    """
    Create bins with approximately equal distribution of samples.
    
    Args:
        labels (np.ndarray): Array of redshift values.
        n_bins (int): Number of bins.
    
    Returns:
        bins (np.ndarray): Adaptive bin edges.
        bin_indices (np.ndarray): Bin indices for each redshift value.
        bin_means (np.ndarray): Mean redshift value for each bin.
    """
    # Calculate quantiles to create adaptive bin edges
    bins = np.quantile(labels, np.linspace(0, 1, n_bins + 1))
    
    # Assign each redshift to a bin
    bin_indices = np.clip(np.digitize(labels, bins) - 1, 0, n_bins - 1)
    
    # Calculate bin means
    bin_means = np.array([labels[bin_indices == i].mean() for i in range(n_bins)])
    
    return bins, bin_indices, bin_means


# Define the CNN for classification
class RedshiftClassifier(nn.Module):
    def __init__(self, n_bins):
        super(RedshiftClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 76x76
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 38x38
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 38 * 38, 128),  # Adjusted input size
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, n_bins)  # Output probabilities for each bin
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x

# Custom Dataset
class GalaxyDataset(Dataset):
    def __init__(self, images, labels, redshifts, transform=None):
        self.images = images
        self.labels = labels
        self.redshifts = redshifts
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        redshift = self.redshifts[idx]
        if self.transform:
            image = self.transform(Image.fromarray(image.astype('uint8')).convert("RGB"))
        return image, torch.tensor(label, dtype=torch.long), redshift  # Labels are class indices for classification

def plot_redshift_distribution(redshifts, bin_indices, n_bins, bin_means):
    """
    Plot the distribution of redshifts after discretization in a histogram.
    
    Args:
        redshifts (np.ndarray): Array of original redshift values.
        bin_indices (np.ndarray): Array of bin indices after discretization.
        n_bins (int): Number of bins used for discretization.
        bin_means (np.ndarray): Array of mean values for each bin.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot original redshift distribution
    plt.hist(redshifts, bins=n_bins, alpha=0.5, label="Original Redshift Distribution", color="blue", edgecolor="black")

    # Plot discretized bin distribution
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    plt.bar(bin_means, bin_counts, width=np.diff(bin_means).mean(), alpha=0.6, color="orange", label="Discretized Bins")

    plt.xlabel("Redshift")
    plt.ylabel("Count")
    plt.title("Redshift Distribution Before and After Discretization")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# Training and Evaluation
if __name__ == "__main__":
    def train_and_evaluate(n_bins=64):
        EPOCHS = 10
        N_SAMPLES = 500
        # Load dataset
        with h5py.File(DATA_DIR, 'r') as f:
            images = np.array(f['images'][:N_SAMPLES])
            labels = np.array(f['redshifts'][:N_SAMPLES])

            print(f"{labels.shape[0]} valid samples loaded")
            print(images.shape)
            print(labels.shape)

            # Create bins for classification
            #bins, bin_indices, bin_means = create_bins(labels, n_bins)
            bins, bin_indices, bin_means = create_adaptive_bins(labels, n_bins)

            #plot_redshift_distribution(labels, bin_indices, n_bins, bin_means)

            # Split the data into training, validation, and testing sets
            images_train, images_temp, labels_train, labels_temp, redshift_train, redshift_temp = train_test_split(
                images, bin_indices, labels, test_size=0.3, random_state=RANDOM_SEED
            )
            images_val, images_test, labels_val, labels_test, redshift_val, redshift_test = train_test_split(
                images_temp, labels_temp, redshift_temp, test_size=0.5, random_state=RANDOM_SEED
            )

        mean = np.mean(images, axis=(0, 1, 2))  # Normalize by 255
        std = np.std(images, axis=(0, 1, 2))

        # Preprocessing and transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        ])

        # Create datasets and dataloaders
        train_dataset = GalaxyDataset(images_train, labels_train, redshift_train, transform=transform)
        val_dataset = GalaxyDataset(images_val, labels_val, redshift_val, transform=transform)
        test_dataset = GalaxyDataset(images_test, labels_test, redshift_test, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model, loss, and optimizer
        model = RedshiftClassifier(n_bins=n_bins).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        start = time.time()
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            val_loss = val_loss / len(val_loader.dataset)
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        end = time.time()
        training_time = end - start
        print(f"Training took {training_time} seconds")

        # Evaluation on the test set
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for inputs, targets, redshifts in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)  # Logits
                probs = torch.softmax(outputs, dim=1)  # Probabilities
                pred_redshift = (probs * torch.tensor(bin_means, device=device)).sum(dim=1)  # Weighted sum
                predictions.extend(pred_redshift.cpu().numpy())
                true_labels.extend(redshifts)

        # Convert true labels (class indices) back to redshift values
        #true_redshifts = [bin_means[label] for label in true_labels]
        true_redshifts = true_labels
        # Compute R² score
        r2 = r2_score(true_redshifts, predictions)
        print(f"R-squared score on test set: {r2:.4f}")

        # Plot predicted vs true redshift
        plt.scatter(true_redshifts, predictions, alpha=0.5)
        plt.plot([min(true_redshifts), max(true_redshifts)],
                [min(true_redshifts), max(true_redshifts)], 'r--')
        plt.xlabel("True Redshift")
        plt.ylabel("Predicted Redshift")
        plt.title(f"Test Set Predictions (R² = {r2:.4f})")
        plt.show()

        return training_time, r2

    training_time, r2 = train_and_evaluate(n_bins=8)
    print(f"Time: {training_time:.2f} seconds, R²: {r2:.4f}")
