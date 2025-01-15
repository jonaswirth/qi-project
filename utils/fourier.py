import numpy as np
import matplotlib.pyplot as plt
import h5py

with h5py.File("../datasets/astroclip_reduced_2.h5") as f:
    spectrum = np.array(f["spectra"][0]).squeeze(axis=-1)

# Compute the FFT of the spectrum
t = range(0, len(spectrum))
spectrum_fft = np.fft.fft(spectrum)

# Compute the frequencies corresponding to the FFT coefficients
frequencies = np.fft.fftfreq(len(spectrum), t[1] - t[0])

# Choose how many Fourier coefficients to keep (e.g., 10 most significant)
num_coeffs_to_keep = 50

# Create a copy of the spectrum_fft and zero out coefficients that are not in the top `num_coeffs_to_keep`
spectrum_fft_truncated = np.copy(spectrum_fft)
spectrum_fft_truncated[num_coeffs_to_keep:-num_coeffs_to_keep] = 0

# Reconstruct the spectrum using the inverse FFT
spectrum_reconstructed = np.fft.ifft(spectrum_fft_truncated)

# Plot the original spectrum and the reconstructed spectrum
plt.figure(figsize=(10, 6))
plt.plot(t, spectrum, label="Original Spectrum")
plt.plot(t, spectrum_reconstructed.real, label="Reconstructed Spectrum", linestyle='--')
plt.xlabel('Time / Wavelength')
plt.ylabel('Amplitude')
plt.legend()
plt.title(f"Fourier Series Approximation with {num_coeffs_to_keep} Terms")
plt.show()
