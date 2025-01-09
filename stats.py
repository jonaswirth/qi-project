import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

classical_stats = pd.read_csv("stats/classic/trainstats_proc.csv", delimiter=";")
quantum_stats = pd.read_csv("stats/hybrid/overall_proc.csv", delimiter=";")

plt.plot(classical_stats["class"], classical_stats["r2"], label="Classical", color="red", linewidth=2)
print(f"Mean r2 classical: {np.mean(classical_stats['r2'])}")

for n_qubit in range(2, 8):
    data = quantum_stats[quantum_stats["n_qubits"] == n_qubit]
    plt.plot(data["class"], data["r2"], label=f"qubits = {n_qubit}")
    print(f"Mean r2 {n_qubit} qubit: {np.mean(data['r2'])}")

ax = plt.gca()
ax.set_ylim([-1, 1])
plt.legend()
plt.plot([0, 9], [0, 0], linestyle="--", color="gray")
plt.show()
