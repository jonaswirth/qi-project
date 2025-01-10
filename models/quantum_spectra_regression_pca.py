import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, ZFeatureMap 
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.primitives import StatevectorEstimator as Estimator
import h5py

import matplotlib.pyplot as plt

RANDOM_STATE = 42

#Create fake data:
#Sample: Random vector with n dimensions
#Label: sum of the components of the vector
def load_and_prepare_fake_data(num_samples, num_dim):
    samples = []
    labels = []
    for i in range(num_samples):
        sample = np.random.rand(num_dim)
        samples.append(sample)
        labels.append(sum(sample))
    
    samples = np.array(samples)
    labels = np.array(labels)

    scaler = MinMaxScaler((-1, 1))
    samples = scaler.fit_transform(samples)

    scaler.fit(labels.reshape(-1,1))
    labels = scaler.transform(labels.reshape(-1,1))
    labels = labels.flatten()
    sample_train, sample_test, label_train, label_test = train_test_split(samples, labels, test_size=0.2, random_state=RANDOM_STATE)
    return sample_train, sample_test, label_train, label_test, scaler


def load_and_prepare_data(num_samples, num_dim):
    with h5py.File("../datasets/astroclip_reduced_2.h5", "r") as f:
        data = np.array(f["spectra"][:num_samples]).squeeze(axis = -1)
        labels = np.array(f["redshifts"][:num_samples])
        
        pca = PCA(n_components=num_dim)
        data = pca.fit_transform(data)

        scaler = MinMaxScaler((-1, 1))
        data = scaler.fit_transform(data)

        scaler.fit(labels.reshape(-1,1))
        labels = scaler.transform(labels.reshape(-1,1))
        labels = labels.flatten()
        sample_train, sample_test, label_train, label_test = train_test_split(data, labels, test_size=0.2, random_state=RANDOM_STATE)
        
    return sample_train, sample_test, label_train, label_test, scaler


if __name__ == "__main__":
    NUM_SAMPLES = 100
    NUM_QUBITS = 5
    sample_train, sample_test, label_train, label_test, scaler = load_and_prepare_data(100, NUM_QUBITS)
    #sample_train, sample_test, label_train, label_test, scaler = load_and_prepare_fake_data(100, NUM_QUBITS)

    original_redshifts = scaler.inverse_transform(label_train.reshape(-1,1)).flatten()

    def callback(weights, obj_func_eval):
        print(obj_func_eval)

    estimator = Estimator()

    featureMap = ZFeatureMap(NUM_QUBITS, reps=3)
    ansatz = RealAmplitudes(NUM_QUBITS, reps=3)

    # qc = QNNCircuit(feature_map = featureMap, ansatz = ansatz)
    # estimator_qnn = EstimatorQNN(circuit=qc, estimator=estimator)

    # regressor = NeuralNetworkRegressor(
    #     neural_network=estimator_qnn,
    #     loss="squared_error",
    #     optimizer=COBYLA(maxiter=100),
    #     callback=callback
    # )

    # regressor.fit(sample_train, label_train)
    # regressor.score(sample_train, label_train)

    # pred = regressor.predict(sample_test)

    vqr = VQR(feature_map=featureMap, ansatz=ansatz, optimizer=COBYLA(maxiter=50), callback=callback, estimator=estimator)
    vqr.fit(sample_train, label_train)
    pred = vqr.predict(sample_test)

    r2 = r2_score(label_test, pred)
    plt.scatter(label_test, pred, alpha=0.5)
    plt.plot([min(label_test), max(label_test)], [min(label_test), max(label_test)], 'r--')
    plt.xlabel("True Redshift")
    plt.ylabel("Predicted Redshift")
    plt.title(f"Predicted vs True Redshift (R² = {r2:.4f})")
    plt.show()






