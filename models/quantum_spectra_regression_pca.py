import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.algorithms.regressors import VQR
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.utils import algorithm_globals
import h5py
import time
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as RealEstimator

import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
algorithm_globals.random_seed = RANDOM_STATE

OPTIMIZER_MAX_ITER = 3
FILE_PATH = "../stats/quantum_spec_regr_gridsearch.csv"

#Load the dataset
def load_data(num_samples):
    with h5py.File("../datasets/astroclip_reduced_2.h5", "r") as f:
        samples = np.array(f["spectra"][:num_samples]).squeeze(axis = -1)
        labels = np.array(f["redshifts"][:num_samples])
    return samples, labels


#Prepare the data: use PCA to reduce the dimension to number of qubits, scale the data to range (-1, 1) and perform train test split
def prepare_data(samples, labels, n_qubits):
    pca = PCA(n_components=n_qubits, random_state=RANDOM_STATE)
    samples = pca.fit_transform(samples)

    scaler = MinMaxScaler((-1, 1))
    samples = scaler.fit_transform(samples)

    scaler.fit(labels.reshape(-1,1))
    labels = scaler.transform(labels.reshape(-1,1))
    labels = labels.flatten()
    sample_train, sample_test, label_train, label_test = train_test_split(samples, labels, test_size=0.2, random_state=RANDOM_STATE)
    return sample_train, sample_test, label_train, label_test, scaler

#Train and evaluate a VQR with given configuration
def train_and_evaluate(samples, labels, n_qubits, reps_feat_map, reps_ansatz):
    start = time.time()
    print(f"Running with config:\nN QUBITS: {n_qubits}\nReps feature map:{reps_feat_map}\nReps ansatz:{reps_ansatz}")

    sample_train, sample_test, label_train, label_test, scaler = prepare_data(samples, labels, n_qubits)

    #Estimator for local simulation:
    estimator = Estimator(seed=RANDOM_STATE)

    featureMap = PauliFeatureMap(n_qubits, reps=reps_feat_map, paulis=['Z'])
    ansatz = RealAmplitudes(n_qubits, reps=reps_ansatz)

    def callback(_, eval):
        print(eval)

    vqr = VQR(feature_map=featureMap, ansatz=ansatz, optimizer=COBYLA(maxiter=OPTIMIZER_MAX_ITER), estimator=estimator, callback=callback)
    vqr.fit(sample_train, label_train)
    pred = vqr.predict(sample_test)

    # Revert scaling
    pred = scaler.inverse_transform(pred.reshape(-1,1)).flatten()
    label_test = scaler.inverse_transform(label_test.reshape(-1,1)).flatten()

    plot(label_test, pred)



def plot(label_test, pred):
    r2 = r2_score(label_test, pred)
    plt.scatter(label_test, pred, alpha=0.5)
    plt.plot([min(label_test), max(label_test)], [min(label_test), max(label_test)], 'r--')
    plt.xlabel("True Redshift")
    plt.ylabel("Predicted Redshift")
    plt.title(f"Predicted vs True Redshift (R² = {r2:.4f})")
    plt.show()
                

#Best config: NUMSAMPLES = 100, NUM_QUBITS = 5, MAXITER = 50
# r2= 0.4102 NUM_SAMPLES = 200, NUM_QUBITS = 5, OPTIMIZER_MAX_ITER = 100, ZFeatureMap reps=4, RealAmplitudes reps 2
if __name__ == "__main__":
    start = time.time()
    
    NUM_SAMPLES = 200
    NUM_QUBITS = 15
    REPS_FEATURE_MAP = 4
    REPS_ANSATZ = 2
    samples, labels = load_data(NUM_SAMPLES)
    train_and_evaluate(samples, labels, NUM_QUBITS, REPS_FEATURE_MAP, REPS_ANSATZ)

    end = time.time()
    print(f"Finished in {end - start} seconds")

    






