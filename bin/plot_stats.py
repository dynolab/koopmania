import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("stats_kt.csv", sep=" ")
plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.plot(df.iloc[:, 0], df.iloc[:, 1], label="DMD")
plt.xlabel("n samples")
plt.ylabel("Train MAPE")
plt.xscale("log")
plt.show()
