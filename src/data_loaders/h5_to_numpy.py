import numpy as np
import h5py

hf = h5py.File("/Users/mariamitrankova/Documents/GitHub/koopmania/src/data_loaders/snapshots/snapshots_s1.h5", "r")
data = hf.get("tasks")['vorticity']
data = data[:]
print(type(data), data.shape)
with open("rb_flow.npy", "wb") as f:
    np.save(f, data)