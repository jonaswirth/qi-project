import torch
import h5py
from sklearn.decomposition import PCA
import numpy as np
from torchvision import transforms
import torch.nn as nn

RANDOM_STATE = 42

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Linear(16, 5)  # Outputs 6-dimensional features
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x

def read_from_file(n_samples):
    with h5py.File("../datasets/astroclip_reduced_3.h5", "r") as f:
        images = np.array(f["images"][:n_samples])
        spectra = np.array(f["spectra"][:n_samples]).squeeze(axis=-1)
        redshifts = np.array(f["redshifts"][:n_samples])

        return images, spectra, redshifts

def process_spectra(spectra):
    pca = PCA(n_components=5, random_state=RANDOM_STATE)
    return pca.fit_transform(spectra)

def encode_images(images):
    encoder = ImageEncoder()
    encoder.load_state_dict(torch.load("../pretrained/image_encoder.pth", map_location=torch.device('cpu')))
    encoder.eval()

    #TODO: Normalize images?
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    features = []
    for img in images:
        img = transform(img).unsqueeze(0)
        img = img.to(torch.float32)
        with torch.no_grad():
            feature = encoder(img)
        features.append(feature.squeeze().numpy())
    
    return features

def store_file(images, spectra, redshifts):
    with h5py.File("../datasets/astroclip_quantum.h5", "w") as f:
        f.create_dataset('images', data=images)
        f.create_dataset('spectra', data = spectra)
        f.create_dataset('redshifts', data=redshifts)
    
if __name__ == "__main__":
    NUM_SAMPLES = 500
    images, spectra, redshifts = read_from_file(NUM_SAMPLES)
    print(f"Images: {images.shape} Spectra: {spectra.shape} Redshifts: {redshifts.shape}")
    spectra = np.array(process_spectra(spectra))
    images = np.array(encode_images(images))
    print(f"Images: {images.shape} Spectra: {spectra.shape} Redshifts: {redshifts.shape}")
    store_file(images, spectra, redshifts)