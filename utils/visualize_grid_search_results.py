import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../stats/quantum_spec_regr_gridsearch.csv", delimiter=";")

print(df.head())

x = range(2, 15)
y21 = np.array(df.loc[(df['Reps featuremap'] == 2) & (df['Reps ansatz'] == 1), 'r2'])
y22 = np.array(df.loc[(df['Reps featuremap'] == 2) & (df['Reps ansatz'] == 2), 'r2'])
y31 = np.array(df.loc[(df['Reps featuremap'] == 3) & (df['Reps ansatz'] == 1), 'r2'])
y32 = np.array(df.loc[(df['Reps featuremap'] == 3) & (df['Reps ansatz'] == 2), 'r2'])

plt.title("R2 score of different configurations")
plt.plot(x, y21, label="Feat. Reps: 2, Ansatz Reps: 1")
plt.plot(x, y22, label="Feat. Reps: 2, Ansatz Reps: 2")
plt.plot(x, y31, label="Feat. Reps: 3, Ansatz Reps: 1")
plt.plot(x, y32, label="Feat. Reps: 3, Ansatz Reps: 2")
plt.legend()
plt.show()
