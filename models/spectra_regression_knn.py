import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
def load_dataset(filepath, num_samples=20000):
    """
    Load the resampled spectra and redshift from the HDF5 file.

    Args:
        filepath (str): Path to the HDF5 file.
        num_samples (int): Number of samples to load.

    Returns:
        tuple: (spectra, redshift)
    """
    with h5py.File(filepath, "r") as f:
        spectra = f["spectra"][:num_samples]
        redshift = f["redshifts"][:num_samples]

        spectra = spectra.squeeze(axis=-1)

    return spectra, redshift

# Train and evaluate KNN
def knn_predict(spectra, redshift, test_size=0.15, val_size=0.15, max_neighbors=20):
    """
    Train and evaluate a KNN regressor for redshift prediction.

    Args:
        spectra (np.ndarray): Resampled spectral data.
        redshift (np.ndarray): Redshift labels.
        test_size (float): Proportion of data to use for testing.
        val_size (float): Proportion of data to use for validation.
        max_neighbors (int): Maximum number of neighbors to consider.

    Returns:
        dict: Contains train, validation, and test performance metrics.
    """
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(spectra, redshift, test_size=test_size + val_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42)

    # Tune k using the validation set
    best_k = 1
    best_r2 = -np.inf
    val_r2_scores = []

    for k in range(1, max_neighbors + 1):
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(X_train, y_train)
        val_predictions = knn.predict(X_val)
        r2 = r2_score(y_val, val_predictions)
        val_r2_scores.append(r2)

        if r2 > best_r2:
            best_r2 = r2
            best_k = k

    print(f"Best k: {best_k}, Validation R²: {best_r2:.4f}")

    # Train final model with the best k
    knn = KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1)
    knn.fit(X_train, y_train)

    # Evaluate on test set
    test_predictions = knn.predict(X_test)
    test_r2 = r2_score(y_test, test_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Return metrics and predictions
    return {
        "best_k": best_k,
        "validation_r2_scores": val_r2_scores,
        "test_r2": test_r2,
        "test_mse": test_mse,
        "test_predictions": test_predictions,
        "true_labels": y_test,
    }

if __name__ == "__main__":
    # Load dataset
    DATASET_PATH = "../datasets/astroclip_reduced_1.h5"
    spectra, redshift = load_dataset(DATASET_PATH, num_samples=20000)

    # Train and evaluate KNN
    print("Training and evaluating KNN regressor...")
    print(spectra.shape)
    results = knn_predict(spectra, redshift, test_size=0.15, val_size=0.15, max_neighbors=64)

    # Plot validation R² scores
    plt.plot(range(1, len(results["validation_r2_scores"]) + 1), results["validation_r2_scores"], marker="o")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Validation R²")
    plt.title("Validation R² vs. Number of Neighbors")
    plt.show()

    # Plot predicted vs true redshift on test set
    plt.scatter(results["true_labels"], results["test_predictions"], alpha=0.5)
    plt.plot([min(results["true_labels"]), max(results["true_labels"])],
             [min(results["true_labels"]), max(results["true_labels"])], 'r--')
    plt.xlabel("True Redshift")
    plt.ylabel("Predicted Redshift")
    plt.title(f"Test Set Predictions (R² = {results['test_r2']:.4f})")
    plt.show()
