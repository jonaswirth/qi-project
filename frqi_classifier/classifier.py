import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler as Sampler

from sklearn.metrics import accuracy_score

def load_dataset():
    with h5py.File("dataset.h5", "r") as f:
        images = np.array(f["images"])
        labels = np.array(f["labels"])
    return images, labels

def train_and_run_model(images, labels):
    images_flat = images.reshape(images.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)

    n_qubits = 4
    sampler = Sampler()
    feature_map = ZZFeatureMap(n_qubits)
    ansatz = RealAmplitudes(n_qubits, reps=1)

    vqc = VQC(feature_map=feature_map, ansatz=ansatz, loss="cross_entropy", optimizer=COBYLA(maxiter=30), sampler=sampler)

    # Train the VQC model
    vqc.fit(X_train, y_train)

    # Evaluate the VQC model
    y_pred = vqc.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    images, labels = load_dataset()
    train_and_run_model()

