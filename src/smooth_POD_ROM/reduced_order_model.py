import numpy as np
from ezyrb import POD, RBF, Database, Linear, RegularGrid
from ezyrb import ReducedOrderModel as ROM


def train_ROM(mu, X, rank):
    db = Database(mu, X.T)
    pod = POD(method="svd", rank=rank)  # = rom.reduction
    reg = RegularGrid()  # = rom.approximation
    rom = ROM(db, pod, reg)
    rom.fit()
    # pod.fit(db.snapshots.T)  # performs SVD
    # reduced_output = pod.transform(db.snapshots.T).T  # transform reduces the given snapshots. = VT.T
    # reg.fit(db.parameters, reduced_output)  # construct interpolators from points and values!
    return rom


def L2_error(X, X_truth):
    L2 = np.mean((X-X_truth)**2, axis=0)**.5
    return L2


def pulse(x, mu, w=0.075+1e-6):
    y = np.zeros_like(x, dtype=np.float64)
    y[(0-1e-6 < (x-mu)) & ((x-mu) < w)] = 1.0
    return y


def delta_n_width(S, a, b):
    delta_n = (np.cumsum(S[::-1]**2)[::-1]/a/b)**.5
    return delta_n
