import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Dataset class
class GalaxyDataset(Dataset):
    def __init__(self, images, spectra, redshifts):
        self.images = images
        self.spectra = spectra
        self.redshifts = redshifts

    def __len__(self):
        return len(self.redshifts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx], dtype=torch.float32),
            torch.tensor(self.spectra[idx], dtype=torch.float32),
            torch.tensor(self.redshifts[idx], dtype=torch.float32),
        )

# Image encoder (simple CNN)
class ImageEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(ImageEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample: 152 -> 76
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Downsample: 76 -> 38
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 38 * 38, embedding_size),  # Adjusted input size
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

# Spectra encoder (simple MLP)
class SpectraEncoder(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(SpectraEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(x)

# Load dataset
def load_dataset(filepath, num_samples=500):
    with h5py.File(filepath, "r") as f:
        images = np.array(f["images"][:num_samples])
        spectra = np.array(f["spectra"][:num_samples])
        redshifts = np.array(f["redshifts"][:num_samples])

        spectra = spectra.squeeze(axis=-1)
    return images, spectra, redshifts

# Training and evaluation
def train_and_evaluate(filepath, num_samples=500, embedding_size=128, knn_neighbors=5):
    # Load and split dataset
    images, spectra, redshifts = load_dataset(filepath, num_samples=num_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        list(zip(images, spectra)), redshifts, test_size=0.2, random_state=RANDOM_SEED
    )

    train_dataset = GalaxyDataset(
        images=[x[0] for x in X_train], spectra=[x[1] for x in X_train], redshifts=y_train
    )
    test_dataset = GalaxyDataset(
        images=[x[0] for x in X_test], spectra=[x[1] for x in X_test], redshifts=y_test
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define encoders
    img_encoder = ImageEncoder(embedding_size).train()
    spec_encoder = SpectraEncoder(input_size=spectra.shape[1], embedding_size=embedding_size).train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(img_encoder.parameters()) + list(spec_encoder.parameters()), lr=0.001)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        img_encoder.train()
        spec_encoder.train()
        epoch_loss = 0.0

        for img, spec, redshift in train_loader:
            optimizer.zero_grad()
            img_embedding = img_encoder(img.permute(0, 3, 1, 2))  # Rearrange channels for Conv2D
            spec_embedding = spec_encoder(spec)
            combined_embedding = (img_embedding + spec_embedding) / 2
            loss = criterion(combined_embedding.mean(dim=1), redshift)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    # Extract embeddings for kNN
    img_encoder.eval()
    spec_encoder.eval()
    train_embeddings, train_redshifts = [], []
    test_embeddings_img, test_redshifts = [], []
    test_embeddings_spec = []

    with torch.no_grad():
        for img, spec, redshift in train_loader:
            img_embedding = img_encoder(img.permute(0, 3, 1, 2))
            train_embeddings.append(img_embedding.cpu().numpy())
            train_redshifts.append(redshift.numpy())
            spec_embedding = spec_encoder(spec)
            train_embeddings.append(spec_embedding.cpu().numpy())
            train_redshifts.append(redshift.numpy())
            #combined_embedding = (img_embedding + spec_embedding) / 2
            #train_embeddings.append(combined_embedding.cpu().numpy())

        for img, spec, redshift in test_loader:
            img_embedding = img_encoder(img.permute(0, 3, 1, 2))
            test_embeddings_img.append(img_embedding.cpu().numpy())

            spec_embedding = spec_encoder(spec)
            test_embeddings_spec.append(spec_embedding.cpu().numpy())

            test_redshifts.append(redshift.numpy())

    train_embeddings = np.vstack(train_embeddings)
    train_redshifts = np.hstack(train_redshifts)
    test_embeddings_img = np.vstack(test_embeddings_img)
    test_embeddings_spec = np.vstack(test_embeddings_spec)
    test_redshifts = np.hstack(test_redshifts)

    # Train kNN regressor
    knn = KNeighborsRegressor(n_neighbors=knn_neighbors)
    knn.fit(train_embeddings, train_redshifts)

    # Predict and evaluate images
    predictions_img = knn.predict(test_embeddings_img)
    r2_img = r2_score(test_redshifts, predictions_img)
    mae_img = mean_absolute_error(test_redshifts, predictions_img)
    print(f"(Image) Test R²: {r2_img:.4f}, Test MAE: {mae_img:.4f}")

    predictions_spec = knn.predict(test_embeddings_spec)
    r2_spec = r2_score(test_redshifts, predictions_spec)
    mae_spec = mean_absolute_error(test_redshifts, predictions_spec)
    print(f"(Spectrum) Test R²: {r2_spec:.4f}, Test MAE: {mae_spec:.4f}")

    # Visualization of image predictions
    plt.scatter(test_redshifts, predictions_img, alpha=0.6)
    plt.plot([min(test_redshifts), max(test_redshifts)], [min(test_redshifts), max(test_redshifts)], color="red")
    plt.xlabel("True Redshift")
    plt.ylabel("Predicted Redshift")
    plt.title(f"True vs. Predicted Redshifts (R²: {r2_img:.4f})")
    plt.show()

    # Visualization of image predictions
    plt.scatter(test_redshifts, predictions_spec, alpha=0.6)
    plt.plot([min(test_redshifts), max(test_redshifts)], [min(test_redshifts), max(test_redshifts)], color="red")
    plt.xlabel("True Redshift")
    plt.ylabel("Predicted Redshift")
    plt.title(f"True vs. Predicted Redshifts (R²: {r2_spec:.4f})")
    plt.show()

# Run the pipeline
if __name__ == "__main__":
    filepath = "../datasets/astroclip_reduced_1.h5"
    train_and_evaluate(filepath, num_samples=1000)
