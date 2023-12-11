import numpy as np
from ezyrb import POD, RBF, Database, Linear, RegularGrid
from ezyrb import ReducedOrderModel as ROM
from smooth_POD_ROM.pre_processing import smoothen
from smooth_POD_ROM.post_processing import post_process


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


def zielfunktion(params, x, mu_train, X_train, mu_test, X_test,
                 rank, shape, num_iter, clip=True):
    dx = x[1] - x[0]
    sigma, c = params[0], params[1]
    print("{:.8f}, {:.8f},".format(sigma, c), end=" ")

    standard_rom = train_ROM(mu_train, X_train, rank=rank)
    X_test_ROM = standard_rom.predict(mu_test).T
    e_ROM = L2_error(X_test_ROM, X_test)
    mean_ROM = np.mean(e_ROM)

    X_train_s = smoothen(X_train, sigma/dx, shape)
    smooth_rom = train_ROM(mu_train, X_train_s, rank=rank)
    X_test_sROM = smooth_rom.predict(mu_test).T
    e_sROM = L2_error(X_test_sROM, X_test)
    mean_sROM = np.mean(e_sROM)
    X_test_sROMs = post_process(x, X_test_sROM, sigma, c, mu_test, mu_train,
                                num_iter, shape=shape, clip=clip)
    e_sROMs = L2_error(X_test_sROMs, X_test)
    mean_sROMs = np.mean(e_sROMs)
    improvement = 100*mean_sROMs/mean_ROM-100
    print("{:.8f}, {:.8f}, {:.8f}, {:.4f} %".format(
        mean_ROM, mean_sROM, mean_sROMs, improvement))
    return improvement, X_test_ROM, X_test_sROM, X_test_sROMs


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
