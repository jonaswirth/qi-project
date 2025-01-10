import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import h5py

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Custom Dataset
class GalaxyDataset(Dataset):
    def __init__(self, images, spectra, labels):
        self.images = images
        self.spectra = spectra
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx], dtype=torch.float32),
            torch.tensor(self.spectra[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

# Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(ImageEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 38 * 38, embedding_size),  # Adjust input size if needed
            nn.ReLU(),
            nn.Linear(embedding_size, 1)  # Regression head for redshift
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

# Spectra Encoder
class SpectraEncoder(nn.Module):
    def __init__(self, input_size, embedding_size):
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
            nn.Linear(64 * (input_size // 8), 128),  # Flatten and pass to dense layer
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

# Training Function
def train_model(image_encoder, spectra_encoder, train_loader, val_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        image_encoder.train()
        spectra_encoder.train()

        total_loss = 0.0

        for img, spec, label in train_loader:
            img, spec, label = img.to(device), spec.to(device), label.to(device)

            optimizer.zero_grad()

            # Forward passes
            img_pred = image_encoder(img.permute(0, 3, 1, 2))  # Image regression
            spec_pred = spectra_encoder(spec)  # Spectrum regression

            # Combined prediction
            combined_pred = (img_pred + spec_pred) / 2

            # Loss
            loss = criterion(combined_pred.squeeze(-1), label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return image_encoder, spectra_encoder

# Evaluation Function
def evaluate_model(image_encoder, spectra_encoder, test_loader, device):
    image_encoder.eval()
    spectra_encoder.eval()

    predictions, true_labels = [], []

    with torch.no_grad():
        for img, spec, label in test_loader:
            img, spec = img.to(device), spec.to(device)

            # Predictions
            img_pred = image_encoder(img.permute(0, 3, 1, 2))
            spec_pred = spectra_encoder(spec)
            combined_pred = (img_pred + spec_pred) / 2

            predictions.extend(combined_pred.squeeze(-1).cpu().numpy())
            true_labels.extend(label.numpy())

    # Compute R²
    r2 = r2_score(true_labels, predictions)
    print(f"Test R²: {r2:.4f}")

    return predictions, true_labels

# Main Function
if __name__ == "__main__":
    # Paths and setup
    DATASET_PATH = "../datasets/astroclip_reduced_2.h5"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 25
    EMBEDDING_SIZE = 128
    NUM_SAMPLES = 500

    # Load dataset
    print("Loading dataset...")
    with h5py.File(DATASET_PATH, "r") as f:
        images = np.array(f["images"][:NUM_SAMPLES])
        spectra = np.array(f["spectra"][:NUM_SAMPLES])
        redshifts = np.array(f["redshifts"][:NUM_SAMPLES])
        spectra = spectra.squeeze(axis=-1)

    dataset = GalaxyDataset(images, spectra, redshifts)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Initialize models, loss, and optimizer
    img_encoder = ImageEncoder(embedding_size=EMBEDDING_SIZE).to(DEVICE)
    spec_encoder = SpectraEncoder(input_size=spectra.shape[1], embedding_size=EMBEDDING_SIZE).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(img_encoder.parameters()) + list(spec_encoder.parameters()),
        lr=0.001
    )
    criterion = nn.MSELoss()

    # Train the model
    print("Training model...")
    train_model(img_encoder, spec_encoder, train_loader, val_loader, optimizer, criterion, EPOCHS, DEVICE)

    # Evaluate the model
    print("Evaluating model...")
    predictions, true_labels = evaluate_model(img_encoder, spec_encoder, test_loader, DEVICE)

    r2 = r2_score(true_labels, predictions)

    # Plot predicted vs true redshift
    plt.scatter(true_labels, predictions, alpha=0.5)
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 'r--')
    plt.xlabel("True Redshift")
    plt.ylabel("Predicted Redshift")
    plt.title(f"Predicted vs True Redshift (R² = {r2:.4f})")
    plt.show()
