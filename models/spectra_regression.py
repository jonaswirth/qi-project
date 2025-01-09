import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define a custom PyTorch dataset
class GalaxySpectraDataset(Dataset):
    def __init__(self, spectra, labels):
        self.spectra = spectra
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        label = self.labels[idx]

        return torch.tensor(spectrum, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Define a simple FNN for regression
class RedshiftCNN(nn.Module):
    def __init__(self, input_dim):
        super(RedshiftCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  # (batch, 1, input_dim) -> (batch, 16, input_dim)
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample by a factor of 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # (batch, 16, input_dim/2) -> (batch, 32, input_dim/2)
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample again
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),  # (batch, 32, input_dim/4) -> (batch, 64, input_dim/4)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample to 1/8 of the input length
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * (input_dim // 8), 128),  # Flatten and pass to dense layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for regression
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, input_dim) -> (batch, 1, input_dim)
        x = self.conv_layers(x)  # Apply convolutional layers
        x = torch.flatten(x, start_dim=1)  # Flatten for fully connected layers
        x = self.fc_layers(x)  # Apply fully connected layers
        return x

# Load the dataset
def load_dataset(filepath, num_samples=500, normalize_redshift=True):
    with h5py.File(filepath, "r") as f:
        spectra = np.array(f["spectra"][:num_samples])
        redshift = np.array(f["redshifts"][:num_samples])
        spectra = spectra.squeeze(axis=-1)

    # Check for NaNs
    spectra = np.nan_to_num(spectra, nan=0.0)
    redshift = np.nan_to_num(redshift, nan=np.mean(redshift))

    # Normalize spectra
    mean_spectrum = np.mean(spectra, axis=0)
    std_spectrum = np.std(spectra, axis=0)
    spectra = (spectra - mean_spectrum) / (std_spectrum + 1e-8)

    return spectra, redshift


# Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device="cpu"):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for spectra, labels in train_loader:
            spectra, labels = spectra.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(spectra).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spectra.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0.0
        predictions, true_labels = [], []

        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra, labels = spectra.to(device), labels.to(device)
                outputs = model(spectra).squeeze()
                val_loss += criterion(outputs, labels).item() * spectra.size(0)
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Compute R-squared on validation set
        r2 = r2_score(true_labels, predictions)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R²: {r2:.4f}")
        #print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# Evaluate the model on the test set
def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for spectra, labels in test_loader:
            spectra, labels = spectra.to(device), labels.to(device)
            outputs = model(spectra).squeeze()
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Compute R-squared
    r2 = r2_score(true_labels, predictions)
    print(f"Test R² score: {r2:.4f}")

    return predictions, true_labels


if __name__ == "__main__":
    # Paths and device setup
    DATASET_PATH = "../datasets/astroclip_reduced_1.h5"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 40

    # Load the dataset
    print("Loading dataset...")
    spectra, redshift = load_dataset(DATASET_PATH, num_samples=500, normalize_redshift=True)
    print(f"Loaded {len(spectra)} samples.")
    # Check for NaNs in spectra and redshift
    print("Number of NaN values in spectra:", np.isnan(spectra).sum())
    print("Number of NaN values in redshift:", np.isnan(redshift).sum())

    spectra = np.nan_to_num(spectra, nan=0.0)


    # Create dataset and splits
    dataset = GalaxySpectraDataset(spectra, redshift)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Initialize model, loss, and optimizer
    input_dim = spectra.shape[1]  # Input dimension = number of spectral features
    model = RedshiftCNN(input_dim=input_dim).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS, device=DEVICE)

    # Evaluate the model on the test set
    print("Evaluating model on test set...")
    predictions, true_labels = evaluate_model(model, test_loader, device=DEVICE)

    # Denormalize predictions and true labels
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Compute final R² score on denormalized data
    final_r2 = r2_score(true_labels, predictions)
    print(f"Final Test R² score (denormalized): {final_r2:.4f}")

    # Plot training and validation losses
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    # Plot predicted vs true redshift
    plt.scatter(true_labels, predictions, alpha=0.5)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 'r--')
    plt.xlabel("True Redshift")
    plt.ylabel("Predicted Redshift")
    plt.title(f"Predicted vs True Redshift (R² = {final_r2:.4f})")
    plt.show()
