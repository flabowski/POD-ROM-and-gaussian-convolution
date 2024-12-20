import numpy as np
from ezyrb import POD, Database, RegularGrid
from ezyrb import ReducedOrderModel as ROM
from smooth_POD_ROM.pre_processing import smoothen_rowwise
from smooth_POD_ROM.post_processing import get_sigma, richardson_lucy
from scipy.optimize import minimize
from scipy.optimize import direct, Bounds


def snapshots(g, x, mu):
    X = np.zeros((len(x), len(mu)))
    for j, mu_j in enumerate(mu):
        y = g(x, mu_j)
        X[:, j] = y
    return X


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


def snapshots_rowwise(g, x, mu):
    X = np.zeros((len(mu), len(x)))
    for j, mu_j in enumerate(mu):
        y = g(x, mu_j)
        X[j] = y
    return X


def get_data_rw(mu, sigma, g, x, **kwargs):
    dx = x[1] - x[0]
    X = snapshots_rowwise(g, x, mu)
    sig = sigma / dx
    X_s = smoothen_rowwise(X, sig, x.shape)
    return X, X_s


def make_data_rw(mu_min, mu_max, n_samples, g, x, sigma, **kwargs):
    mu = np.linspace(mu_min, mu_max, n_samples, endpoint=False)
    X, X_s = get_data_rw(mu, sigma, g, x, **kwargs)
    return mu, X, X_s


# def make_rand_data_rw(mu_min, mu_max, n_samples, g, x, sigma, **kwargs):
#     mu = np.atleast_2d(np.random.rand(n_samples)[:, None] * (mu_max - mu_min) + mu_min)
#     X, X_s = get_data_rw(mu, sigma, g, x, **kwargs)
#     return mu, X, X_s


def train_ROM_rw(mu, X, rank=False):
    if not rank:
        rank = len(mu)  # no model truncation
    db = Database(mu, X)
    pod = POD(method="svd", rank=rank)  # = rom.reduction
    reg = RegularGrid()  # = rom.approximation
    rom = ROM(db, pod, reg)
    rom.fit()
    # pod.fit(db.snapshots.T)  # performs SVD
    # reduced_output = pod.transform(db.snapshots.T).T  # transform reduces the given snapshots. = VT.T
    # reg.fit(db.parameters, reduced_output)  # construct interpolators from points and values!
    return rom


def L2_error(X, X_truth):
    L2 = np.mean((X - X_truth) ** 2, axis=0) ** 0.5
    return L2


def delta_n_width(S, a, b):
    delta_n = (np.cumsum(S[::-1] ** 2)[::-1] / a / b) ** 0.5
    return delta_n


def L2_error_rw(X, X_truth, axis=1):
    L2 = np.mean((X - X_truth) ** 2, axis=axis) ** 0.5
    return L2


def get_predictions(
    sROM,
    mu_,
    sigma,
    c,
    num_iter,
    shape,
    x,
    monitor_progress_postprocessing=False,
    monitor_convergence=False,
    sigmaD="calc_based_on_distance",
    **kwargs
):
    X_test_sROM = sROM.predict(mu_).snapshots_matrix
    mu_train = sROM.database.parameters_matrix
    data = X_test_sROM

    # from datetime import datetime
    if monitor_convergence:
        deconvolved = np.ones((*data.shape, num_iter))
    else:
        deconvolved = np.empty_like(data)
    for j in range(len(mu_)):
        if sigmaD == "calc_based_on_distance":
            sgm_est = get_sigma(sigma, mu_[j][None, ...], mu_train, c=c)
        else:
            sgm_est = sigmaD[j]
        # t1 = datetime.now()
        deconvolved[j] = richardson_lucy(
            x,
            data[j].reshape(shape),
            sgm_est,
            num_iter,
            monitor_convergence=monitor_convergence,
        ).reshape(deconvolved[j].shape)
        # print("sgm=", sgm_est*1000, (datetime.now()-t1).total_seconds())
        if monitor_progress_postprocessing:
            print(j, end=", ")
    X_test_sROMs = deconvolved
    return X_test_sROM, X_test_sROMs


def get_improvement(X_test, X_test_ROM, X_test_sROM, X_test_sROMs):
    e_ROM = L2_error_rw(X_test_ROM, X_test)
    mean_ROM = np.mean(e_ROM)
    e_sROM = L2_error_rw(X_test_sROM, X_test)
    mean_sROM = np.mean(e_sROM)
    e_sROMs = L2_error_rw(X_test_sROMs, X_test)
    mean_sROMs = np.mean(e_sROMs)
    improvement = 100 * mean_sROMs / mean_ROM - 100
    return mean_ROM, mean_sROM, mean_sROMs, improvement


def target_function(case):
    if ("n_train" in case.keys()) and ("mu_train" in case.keys()):
        raise ValueError("training data is ambiguous, 'n_train' and 'mu_train' are given.")
    if ("n_test" in case.keys()) and ("mu_test" in case.keys()):
        raise ValueError("testing data is ambiguous, 'n_test' and 'mu_test' are given.")

    if "n_train" in case.keys():
        mu_train, X_train, X_train_s = make_data_rw([0], [1], n_samples=case["n_train"], **case)
    elif "mu_train" in case.keys():
        mu_train = case["mu_train"]
        X_train, X_train_s = get_data_rw(mu_train, **case)

    if "n_test" in case.keys():
        mu_test, X_test, X_test_s = make_data_rw(
            mu_train[1], mu_train[2], n_samples=case["n_test"], **case
        )
    elif "mu_test" in case.keys():
        mu_test = case["mu_test"]
        X_test, X_test_s = get_data_rw(mu_test, **case)

    my_ROM = train_ROM_rw(mu_train, X_train)
    my_sROM = train_ROM_rw(mu_train, X_train_s)

    X_test_ROM = my_ROM.predict(mu_test).snapshots_matrix
    X_test_sROM, X_test_sROMs = get_predictions(my_sROM, mu_test, **case)
    mean_ROM, mean_sROM, mean_sROMs, improvement = get_improvement(
        X_test, X_test_ROM, X_test_sROM, X_test_sROMs
    )
    print(
        "{:.0f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.4f}".format(
            case["num_iter"],
            case["sigma"],
            case["c"],
            mean_ROM,
            mean_sROM,
            mean_sROMs,
            improvement,
        )
    )
    return mean_sROMs


def optimize_hyperparameters(case):
    rank = case["rank"]
    sigma_opt = 1 / (rank * 2)

    def target(params):
        case["sigma"], case["c"] = params[0], params[1]
        return target_function(case)

    # x0 = np.array([sigma_opt, 1.0])
    # res = minimize(
    #     target,
    #     x0,
    #     method="SLSQP",
    #     bounds=[(0.001, 5 * sigma_opt), (0, 10)],
    #     options={"disp": True, "eps": np.array([0.0005, 0.05]), "maxiter": 25, "ftol": 0.0005},
    # )
    bounds = Bounds([0.001, 0], [5 * sigma_opt, 10])
    res = direct(target, bounds, eps=1e-2, len_tol=0.025)  # max side length_abs=[0.006, 0.25]
    return res["x"]
