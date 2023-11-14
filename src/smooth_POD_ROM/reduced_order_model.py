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


def L2_eror(X, X_truth):
    L2 = np.mean((X-X_truth)**2, axis=0)**.5
    return L2
