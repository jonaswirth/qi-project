import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torchvision import transforms

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define a custom PyTorch dataset
class GalaxySpectraDataset(Dataset):
    def __init__(self, images, spectra, labels):
        self.images = images
        self.spectra = spectra
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        spectrum = self.spectra[idx]
        label = self.labels[idx]

        return torch.tensor(image, dtype=torch.Float32), torch.tensor(spectrum, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Define the image encoder
class ImageEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(ImageEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 38 * 38, embedding_size),  # Adjust based on input size
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

# Define Spectra Encoder
class SpectraEncoder(nn.Module):
    def __init__(self, input_dim, embedding_size):
        super(SpectraEncoder, self).__init__()
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
            nn.Linear(128, embedding_size)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch, input_dim) -> (batch, 1, input_dim)
        x = self.conv_layers(x)  # Apply convolutional layers
        x = torch.flatten(x, start_dim=1)  # Flatten for fully connected layers
        x = self.fc_layers(x)  # Apply fully connected layers
        return x

#Predict the redshift from the Embeddings
class EmbeddingRegressor(nn.Module):
    def __init__(self, embedding_size):
        super(EmbeddingRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output single scalar for redshift
        )

    def forward(self, embedding):
        return self.fc(embedding)

# Load the dataset
def load_dataset(filepath, num_samples=500, normalize_redshift=True):
    with h5py.File(filepath, "r") as f:
        images = np.array(f["images"][:num_samples])
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

    #TODO: normalize images
    # mean = np.mean(images, axis=(0, 1, 2)) / 255.0  # Normalize by 255
    # std = np.std(images, axis=(0, 1, 2)) / 255.0

    # # Preprocessing and transforms
    # transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    #     ])

    return images, spectra, redshift


# Train the model
def train_model(imgEncoder, specEncoder, regressor, train_loader, val_loader, criterion, optimizer, epochs=10, device="cpu"):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        specEncoder.train()
        running_loss = 0.0

        for img, spectra, labels in train_loader:
            spectra, labels = spectra.to(device), labels.to(device)

            optimizer.zero_grad()

            img_embedding = imgEncoder(img)
            spec_embedding = specEncoder(spectra)
            
            redshift = regressor(spec_embedding).squeeze()

            loss = criterion(redshift.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spectra.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation step
        specEncoder.eval()
        val_loss = 0.0
        predictions, true_labels = [], []

        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra, labels = spectra.to(device), labels.to(device)
                spec_embedding = specEncoder(spectra)
                redshift = regressor(spec_embedding).squeeze()

                val_loss += criterion(redshift, labels).item() * spectra.size(0)
                predictions.extend(redshift.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Compute R-squared on validation set
        r2 = r2_score(true_labels, predictions)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R²: {r2:.4f}")
        #print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# Evaluate the model on the test set
def evaluate_model(imgEncoder, specEncoder, regressor, test_loader, device="cpu"):
    specEncoder.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for img, spectra, labels in test_loader:
            spectra, labels = spectra.to(device), labels.to(device)
            
            spec_embeddings = specEncoder(spectra).squeeze()
            img_embeddings = imgEncoder(img)

            outputs = regressor(spec_embeddings)

            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Compute R-squared
    r2 = r2_score(true_labels, predictions)
    print(f"Test R² score: {r2:.4f}")

    return predictions, true_labels


if __name__ == "__main__":
    # Paths and device setup
    DATASET_PATH = "../datasets/astroclip_reduced_2.h5"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 40
    EMBEDDING_SIZE = 64

    # Load the dataset
    print("Loading dataset...")
    spectra, redshift = load_dataset(DATASET_PATH, num_samples=100, normalize_redshift=True)
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
    imgEncoder = ImageEncoder(embedding_size=EMBEDDING_SIZE).to(DEVICE)
    specEncoder = SpectraEncoder(input_dim=input_dim, embedding_size=EMBEDDING_SIZE).to(DEVICE)
    regressor = EmbeddingRegressor(embedding_size=EMBEDDING_SIZE).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(specEncoder.parameters(), lr=0.001)

    # Train the model
    print("Training model...")
    train_losses, val_losses = train_model(imgEncoder, specEncoder, regressor, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS, device=DEVICE)

    # Evaluate the model on the test set
    print("Evaluating model on test set...")
    predictions, true_labels = evaluate_model(imgEncoder, specEncoder, regressor, test_loader, device=DEVICE)

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
