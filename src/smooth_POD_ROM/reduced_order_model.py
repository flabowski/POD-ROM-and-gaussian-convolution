import numpy as np
from ezyrb import POD, RBF, Database, Linear, RegularGrid
from ezyrb import ReducedOrderModel as ROM
from smooth_POD_ROM.pre_processing import smoothen, smoothen_rowwise
from smooth_POD_ROM.post_processing import post_process, get_sigma, richardson_lucy
from scipy.optimize import minimize
from scipy.optimize import direct, Bounds
from warnings import warn
from smooth_POD_ROM.initial_conditions import (
    rectangular_pulse,
    rect_pulse_sin,
    saw_tooth,
    triangle,
)


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


def make_data_rw(mu_min, mu_max, n_samples, g, x, sigma, **kwargs):
    dx = x[1] - x[0]
    mu = np.linspace(mu_min, mu_max, n_samples, endpoint=False)
    X = snapshots_rowwise(g, x, mu)
    sig = sigma / dx
    X_s = smoothen_rowwise(X, sig, x.shape)
    return mu, X, X_s


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


def get_predictions(
    sROM,
    mu_test,
    sigma,
    c,
    num_iter,
    shape,
    x,
    monitor_progress_postprocessing=False,
    monitor_convergence=False,
    **kwargs
):
    X_test_sROM = sROM.predict(mu_test).snapshots_matrix
    mu_train = sROM.database.parameters_matrix
    data = X_test_sROM

    # from datetime import datetime
    if monitor_convergence:
        deconvolved = np.ones((*data.shape, num_iter))
    else:
        deconvolved = np.empty_like(data)
    for j in range(len(mu_test)):
        sgm_est = get_sigma(sigma, mu_test[j][None, ...], mu_train, c=c)
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


# def zielfunktion(
#     params,
#     x,
#     mu_train,
#     X_train,
#     mu_test,
#     X_test,
#     rank,
#     shape,
#     num_iter,
#     clip=True,
#     counter=np.array([0]),
#     maxiter=150,
# ):
#     if counter[0] > maxiter:
#         counter[0] += 1
#         return 0.0, 0.0, 0.0, 0.0
#     dx = x[1] - x[0]
#     sigma, c = params[0], params[1]

#     standard_rom = train_ROM(mu_train, X_train, rank=rank)
#     X_test_ROM = standard_rom.predict(mu_test).snapshots_matrix.T
#     e_ROM = L2_error(X_test_ROM, X_test)
#     mean_ROM = np.mean(e_ROM)

#     X_train_s = smoothen(X_train, sigma / dx, shape)
#     smooth_rom = train_ROM(mu_train, X_train_s, rank=rank)
#     X_test_sROM = smooth_rom.predict(mu_test).snapshots_matrix.T
#     e_sROM = L2_error(X_test_sROM, X_test)
#     mean_sROM = np.mean(e_sROM)
#     X_test_sROMs = post_process(
#         x, X_test_sROM, sigma, c, mu_test, mu_train, num_iter, shape=shape, clip=clip
#     )
#     e_sROMs = L2_error(X_test_sROMs, X_test)
#     mean_sROMs = np.mean(e_sROMs)
#     improvement = 100 * mean_sROMs / mean_ROM - 100
#     print(
#         "{:.8f}, {:.8f},".format(sigma, c),
#         "{:.0f}, {:.8f}, {:.8f}, {:.8f}, {:.4f} %".format(
#             counter[0], mean_ROM, mean_sROM, mean_sROMs, improvement
#         ),
#     )
#     counter[0] += 1
#     return improvement, X_test_ROM, X_test_sROM, X_test_sROMs


# def optimize_hyperparameters(x, mu_train, mu_test, X_train, X_test, shape, num_iter2, clip, rank):
#     sigma_opt = 1 / (rank * 2)
#     x0 = np.array([sigma_opt, 1.0])
#     counter = np.array([0])

#     def zf(params):
#         return zielfunktion(
#             params,
#             x,
#             mu_train,
#             X_train,
#             mu_test,
#             X_test,
#             rank,
#             shape,
#             num_iter2,
#             clip=clip,
#             counter=counter,
#         )[0]

#     print("sigma, c, iteration count, mean_ROM, mean_sROM, mean_sROMs, improvement")
#     res = minimize(
#         zf,
#         x0,
#         method="SLSQP",
#         bounds=[(0.001, 5 * sigma_opt), (0, 10)],
#         options={"disp": True, "eps": np.array([0.0005, 0.05]), "maxiter": 25, "ftol": 0.0005},
#     )
#     # _i, _R, _sR, _sRs = zielfunktion(
#     #     res["x"], x, mu_train, X_train, mu_test, X_test, rank, shape, num_iter2, clip=clip)
#     # n_iter = 1000, sig_opt = 0.06579520, c_opt = 1.13641229, improvement = -46.5767 %
#     # n_iter = 500, sig_opt = 0.06784999, c_opt = 1.05239696, improvement = -44.7313 % or -43.3101?
#     return res["x"]


# def optimize_hyperparameters_single(
#     x, mu_train, mu_test, X_train, X_test, shape, num_iter2, clip, rank
# ):

#     sigma_opt = 1 / (rank * 2)
#     x0 = np.array([sigma_opt, 1.0])
#     counter = np.array([0])

#     def zf(params):
#         return zielfunktion(
#             params,
#             x,
#             mu_train,
#             X_train,
#             mu_test,
#             X_test,
#             rank,
#             shape,
#             num_iter2,
#             clip=clip,
#             counter=counter,
#         )[0]

#     print("sigma, c, iteration count, mean_ROM, mean_sROM, mean_sROMs, improvement")
#     res = minimize(
#         zf,
#         x0,
#         method="SLSQP",
#         bounds=[(0.001, 10 * sigma_opt), (0, 10)],
#         options={"disp": True, "eps": np.array([0.0005, 0.05]), "maxiter": 25, "ftol": 0.0005},
#     )
#     # _i, _R, _sR, _sRs = zielfunktion(
#     #     res["x"], x, mu_train, X_train, mu_test, X_test, rank, shape, num_iter2, clip=clip)
#     # n_iter = 1000, sig_opt = 0.06579520, c_opt = 1.13641229, improvement = -46.5767 %
#     # n_iter = 500, sig_opt = 0.06784999, c_opt = 1.05239696, improvement = -44.7313 % or -43.3101?
#     return res["x"]


def L2_error(X, X_truth):
    L2 = np.mean((X - X_truth) ** 2, axis=0) ** 0.5
    return L2


def delta_n_width(S, a, b):
    delta_n = (np.cumsum(S[::-1] ** 2)[::-1] / a / b) ** 0.5
    return delta_n


def L2_error_rw(X, X_truth, axis=1):
    L2 = np.mean((X - X_truth) ** 2, axis=axis) ** 0.5
    return L2


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
    mu_train, X_train, X_train_s = make_data_rw([0], [1], n_samples=case["n_train"], **case)
    mu_test, X_test, X_test_s = make_data_rw(
        mu_train[1], mu_train[2], n_samples=case["n_test"], **case
    )

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
    return improvement


def optimize_hyperparameters(case):
    rank = case["n_train"]
    sigma_opt = 1 / (rank * 2)
    x0 = np.array([sigma_opt, 1.0])
    counter = np.array([0])

    def target(params):
        case["sigma"], case["c"] = params[0], params[1]
        return target_function(case)

    # res = minimize(
    #     target,
    #     x0,
    #     method="SLSQP",
    #     bounds=[(0.001, 5 * sigma_opt), (0, 10)],
    #     options={"disp": True, "eps": np.array([0.0005, 0.05]), "maxiter": 25, "ftol": 0.0005},
    # )
    bounds = Bounds([0.001, 0], [5 * sigma_opt, 10])
    res = result = direct(
        target, bounds, eps=1e-2, len_tol=0.025
    )  # max side length_abs=[0.006, 0.25]
    return res["x"]
