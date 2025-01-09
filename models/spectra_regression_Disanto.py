import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import h5py
import matplotlib.pyplot as plt

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define the dataset class
class GalaxySpectraDataset(Dataset):
    def __init__(self, spectra, redshifts):
        """
        Custom Dataset for Galaxy Spectra and Redshift.

        Args:
            spectra (np.ndarray): Input spectra, shape (N, features).
            redshifts (np.ndarray): Target redshift values, shape (N,).
        """
        self.spectra = spectra
        self.redshifts = redshifts

    def __len__(self):
        return len(self.redshifts)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        redshift = self.redshifts[idx]
        return torch.tensor(spectrum, dtype=torch.float32), torch.tensor(redshift, dtype=torch.float32)

# Define the fully connected neural network
class RedshiftDNN(nn.Module):
    def __init__(self, input_size):
        """
        Fully connected neural network for redshift estimation.

        Args:
            input_size (int): Number of input features (spectral bins).
        """
        super(RedshiftDNN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for regression
        )

    def forward(self, x):
        return self.fc_layers(x)

# Load Galaxy10_with_spectra dataset
def load_dataset(filepath, num_samples=500):
    """
    Load the Galaxy10_with_spectra dataset.

    Args:
        filepath (str): Path to the HDF5 file.
        num_samples (int): Limit on the number of samples for testing.

    Returns:
        spectra (np.ndarray): Spectral data.
        redshifts (np.ndarray): Redshift values.
    """
    with h5py.File(filepath, "r") as f:
        spectra = np.array(f["spectra"][:num_samples])
        redshifts = np.array(f["redshifts"][:num_samples])
        spectra = spectra.squeeze(axis=-1)
    return spectra, redshifts

# Initialize model weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Main training loop
def train_model(filepath, batch_size=32, epochs=50, learning_rate=0.0001, num_samples=500):
    """
    Train a fully connected neural network for redshift estimation.

    Args:
        filepath (str): Path to the HDF5 dataset file.
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimization.
        num_samples (int): Limit on the number of samples for testing.
    """
    # Load data
    spectra, redshifts = load_dataset(filepath, num_samples=num_samples)

    # Replace NaN and Inf values
    spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize the spectra
    epsilon = 1e-8
    spectra = (spectra - np.mean(spectra, axis=0)) / (np.std(spectra, axis=0) + epsilon)

    # Normalize redshifts
    min_redshift, max_redshift = np.min(redshifts), np.max(redshifts)
    if max_redshift - min_redshift > 0:
        redshifts = (redshifts - min_redshift) / (max_redshift - min_redshift)
    else:
        raise ValueError("Invalid redshift range: normalization failed.")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(spectra, redshifts, test_size=0.3, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)

    # Create datasets and dataloaders
    train_dataset = GalaxySpectraDataset(X_train, y_train)
    val_dataset = GalaxySpectraDataset(X_val, y_val)
    test_dataset = GalaxySpectraDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    input_size = spectra.shape[1]
    model = RedshiftDNN(input_size)
    model.apply(initialize_weights)  # Initialize weights
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    training_loss_history = []
    validation_loss_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        # Store loss history
        training_loss_history.append(train_loss)
        validation_loss_history.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

    # Plot loss history
    plt.plot(training_loss_history, label="Training Loss")
    plt.plot(validation_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    # Test the model
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().tolist())
            true_labels.extend(targets.tolist())

    # Rescale predictions and true labels to original scale
    predictions = np.array(predictions) * (max_redshift - min_redshift) + min_redshift
    true_labels = np.array(true_labels) * (max_redshift - min_redshift) + min_redshift

    # Calculate R-squared score
    r2 = r2_score(true_labels, predictions)
    print(f"R-squared score on test set: {r2:.4f}")

    # Visualization of image predictions
    plt.scatter(true_labels, predictions, alpha=0.6)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], color="red")
    plt.xlabel("True Redshift")
    plt.ylabel("Predicted Redshift")
    plt.title(f"True vs. Predicted Redshifts (R²: {r2:.4f})")
    plt.show()

    # Save model
    # torch.save(model.state_dict(), "redshift_model.pth")
    # print("Model saved as redshift_model.pth")

# Run the training
if __name__ == "__main__":
    filepath = "../datasets/astroclip_reduced_1.h5"
    train_model(filepath, num_samples=20000)
