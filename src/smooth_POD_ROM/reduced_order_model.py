import numpy as np
from ezyrb import POD, Database, RegularGrid
from ezyrb import ReducedOrderModel as ROM
from smooth_POD_ROM.pre_processing import smoothen_rowwise
from smooth_POD_ROM.post_processing import get_sigma, richardson_lucy, richardson_lucy_gpu, richardson_lucy2_gpu, get_sigma_batch
from scipy.optimize import minimize
from scipy.optimize import direct, Bounds

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_gpu
except ImportError:
    cp = None
    gaussian_filter_gpu = None


def make_training_data(case):
    mu_min, mu_max = [0], [1]
    n_samples = case["n_train"]
    # sigma, g, x = case["sigma"], case["g"], case["x"]
    mu = np.linspace(mu_min, mu_max, n_samples, endpoint=False)
    X, X_s = get_data_rw(mu, **case)
    return mu, X, X_s


def make_test_data(case):
    mu_min, mu_max = [0], [1 - 1 / case["n_train"]]
    n_samples = case["n_test"]
    # sigma, g, x = case["sigma"], case["g"], case["x"]
    mu = np.atleast_2d(np.random.rand(n_samples)[:, None] * (mu_max[0] - mu_min[0]) + mu_min[0])
    X, X_s = get_data_rw(mu, **case)
    return mu, X, X_s


def get_data_rw(mu, sigma, g, x, **kwargs):
    dx = x[1] - x[0]
    X = snapshots_rowwise(g, x, mu)
    sig = sigma / dx
    truncate = 8
    if "truncate" in kwargs.keys():
        truncate = kwargs["truncate"]
    mode = "wrap"
    if "mode" in kwargs.keys():
        mode = kwargs["mode"]
    X_s = smoothen_rowwise(X, sig, x.shape, truncate=truncate, mode=mode)
    return X, X_s


def snapshots_rowwise(g, x, mu):
    X = np.zeros((len(mu), len(x)))
    for j, mu_j in enumerate(mu):
        y = g(x, mu_j)
        X[j] = y
    return X


def L1_error_rw(X, X_truth, axis=1):
    # print("L1")
    L1 = np.mean(np.abs(X - X_truth), axis=axis)
    return L1


def L2_error_rw(X, X_truth, axis=1):
    # print("L2")
    L2 = np.mean((X - X_truth) ** 2, axis=axis) ** 0.5
    return L2


def get_predictions(sROM, mu_, sigma, c, num_iter, shape, x, sROM_only=False, monitor_progress_postprocessing=False, monitor_convergence=False, sigmaD="calc_based_on_distance", **kwargs):
    mu_ = np.asarray(mu_)
    X_test_sROM = sROM.predict(mu_).snapshots_matrix
    mu_train = sROM.database.parameters_matrix
    data = X_test_sROM
    if sROM_only:
        return data, None
    mode = kwargs.get("mode", "wrap")

    # from datetime import datetime
    if monitor_convergence:
        deconvolved = np.ones((*data.shape, num_iter))
    else:
        deconvolved = np.empty_like(data)
    for j in range(len(mu_)):
        # TODO: make sigmaD a callable or somethign..
        if isinstance(sigmaD, np.ndarray):
            sgm_est = sigmaD[j]
        elif sigmaD == "calc_based_on_distance":
            sgm_est = get_sigma(sigma, mu_[j][None, ...], mu_train, c=c)
        else:
            raise ValueError("unknown method for sigmaD.")
        # t1 = datetime.now()
        res = richardson_lucy(
            x,
            data[j].reshape(shape),
            sgm_est,
            num_iter,
            mode=mode,
            monitor_convergence=monitor_convergence,
        )[
            0
        ].reshape(deconvolved[j].shape)
        deconvolved[j] = res
        # print("sgm=", sgm_est*1000, (datetime.now()-t1).total_seconds())
        if monitor_progress_postprocessing:
            print(j, end=", ")
    X_test_sROMs = deconvolved
    return X_test_sROM, X_test_sROMs


def _smooth_gpu2_compute(imgs, sgm, truncate, mode):
    return gaussian_filter_gpu(imgs, sigma=(0, sgm, sgm), truncate=truncate, mode=mode)


def smooth_snapshots_gpu2(X, case):
    shape = case["shape"]
    sgm = case["sigma"] / case["dx"]
    truncate = case["truncate"]
    imgs = cp.asarray(X.reshape(-1, *shape).astype(np.float32))
    mode = case["mode"]
    out = _smooth_gpu2_compute(imgs, sgm, truncate, mode)
    return cp.asnumpy(out).reshape(len(X), -1)


def get_predictions_gpu(sROM, mu_, sigma, c, num_iter, shape, x, sROM_only=False, monitor_progress_postprocessing=False, monitor_convergence=False, sigmaD="calc_based_on_distance", **kwargs):
    mu_ = np.asarray(mu_)
    X_test_sROM = sROM.predict(mu_).snapshots_matrix
    mu_train = sROM.database.parameters_matrix
    data = X_test_sROM
    if sROM_only:
        return data, None
    mode = kwargs.get("mode", "wrap")
    cmax = kwargs.get("cmax", np.ones((len(mu_),)))

    if monitor_convergence:
        raise ValueError("use CPU version")
    # Transfer all data to GPU once
    data_gpu = cp.asarray(data.astype(np.float32))
    deconvolved_gpu = cp.empty_like(data_gpu)

    if isinstance(sigmaD, np.ndarray):
        sgm_est = sigmaD
    elif sigmaD == "calc_based_on_distance":
        sgm_est = get_sigma_batch(sigma, mu_, mu_train, c=c)
    else:
        raise ValueError("unknown method for sigmaD.")

    for j in range(len(mu_)):
        res = richardson_lucy2_gpu(x, data_gpu[j].reshape(shape), sgm_est[j], num_iter, mode=mode, clip_max=cmax[j])[0].reshape(deconvolved_gpu[j].shape)
        deconvolved_gpu[j] = res
    X_test_sROMs = cp.asnumpy(deconvolved_gpu)
    return X_test_sROM, X_test_sROMs


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
        mu_test, X_test, X_test_s = make_data_rw(mu_train[1], mu_train[2], n_samples=case["n_test"], **case)
    elif "mu_test" in case.keys():
        mu_test = case["mu_test"]
        X_test, X_test_s = get_data_rw(mu_test, **case)

    norm = case.get("norm", L2_error_rw)
    scaler = case.get("scaler", None)
    my_ROM = train_ROM_rw(mu_train, X_train)
    my_sROM = train_ROM_rw(mu_train, X_train_s)

    X_test_ROM = my_ROM.predict(mu_test).snapshots_matrix
    X_test_sROM, X_test_sROMs = get_predictions(my_sROM, mu_test, **case)
    mean_ROM, mean_sROM, mean_sROMs, improvement = get_improvement(X_test, X_test_ROM, X_test_sROM, X_test_sROMs, norm=norm, scaler=scaler)
    print("num_iter, n_train, n_test, sigma_S, sigma_D/c, mean_ROM, mean_sROM, mean_sROMs, improvement")
    if "sROM_only" in case.keys():
        if case["sROM_only"]:
            print(
                "{:.0f}, {:.0f}, {:.0f}, {:.8f}, -, {:.8f}, {:.8f}, -, {:.4f}".format(
                    case["num_iter"],
                    len(mu_train),
                    len(mu_test),
                    case["sigma"],
                    mean_ROM,
                    mean_sROM,
                    improvement,
                )
            )
            return mean_ROM, mean_sROM, mean_sROMs
    print(
        "{:.0f}, {:.0f}, {:.0f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.4f}".format(
            case["num_iter"],
            len(mu_train),
            len(mu_test),
            case["sigma"],
            case["c"],
            mean_ROM,
            mean_sROM,
            mean_sROMs,
            improvement,
        )
    )
    return mean_ROM, mean_sROM, mean_sROMs


def optimize_hyperparameters(case):
    rank = case["rank"]
    # sigma_opt = 1 / (rank * 2)
    sigma_opt = 0.2 / rank  # optimal for saw tooth w=1/45

    def target(params):
        case["sigma"], case["c"] = params[0], params[1]
        return target_function(case)[2]

    # x0 = np.array([sigma_opt, 1.0])
    # res = minimize(
    #     target,
    #     x0,
    #     method="SLSQP",
    #     bounds=[(0.001, 5 * sigma_opt), (0, 10)],
    #     options={"disp": True, "eps": np.array([0.0005, 0.05]), "maxiter": 25, "ftol": 0.0005},
    # )
    bounds = Bounds([0.001, 0], [5 * sigma_opt, 2])
    res = direct(target, bounds, eps=1e-2, len_tol=0.025)  # max side length_abs=[0.006, 0.25]
    print(
        "optimization result for sigma:",
        res["x"][0],
        "bounds",
        0.001,
        5 * sigma_opt,
        "guess:",
        sigma_opt,
    )
    print("optimization result for c:", res["x"][1], "bounds", 0, 2)
    print(res.message)
    # TODO: check if bounds were OK
    return res["x"]


def get_improvement(X_test, X_test_ROM, X_test_sROM, X_test_sROMs, norm, scaler=False):
    if scaler:
        raise ValueError("scale before passing to this function")
        # X_test = scaler.scale_up(X_test)
        # X_test_ROM = scaler.scale_up(X_test_ROM)
        # X_test_sROM = scaler.scale_up(X_test_sROM)
        # X_test_sROMs = scaler.scale_up(X_test_sROMs)
    e_ROM = norm(X_test_ROM, X_test)
    mean_ROM = np.mean(e_ROM)
    e_sROM = norm(X_test_sROM, X_test)
    mean_sROM = np.mean(e_sROM)
    if isinstance(X_test_sROMs, np.ndarray):
        e_sROMs = norm(X_test_sROMs, X_test)
        mean_sROMs = np.mean(e_sROMs)
        improvement = 100 * mean_sROMs / mean_ROM - 100
    else:  # sROM_only
        improvement = 100 * mean_sROM / mean_ROM - 100
        mean_sROMs = None
    return mean_ROM, mean_sROM, mean_sROMs, improvement


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


def optimize_sROM(case):
    rank = case["rank"]
    sigma_opt = 0.1 / (rank)

    def target(params):
        case["sigma"] = params[0]
        return target_function(case)[1]

    print("max:", 2 * sigma_opt)
    bounds = Bounds([0.00001], [2 * sigma_opt])
    res = direct(target, bounds, eps=1e-3, len_tol=0.0025)  # max side length_abs=[0.006, 0.25]
    print("optim. res", case["rank"], sigma_opt, 5 * sigma_opt, res["x"])
    print(res.message)
    # TODO: check if bounds were OK
    return res["x"]


# old:
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


def make_data_rw(mu_min, mu_max, n_samples, g, x, sigma, **kwargs):
    mu = np.linspace(mu_min, mu_max, n_samples, endpoint=False)
    X, X_s = get_data_rw(mu, sigma, g, x, **kwargs)
    return mu, X, X_s


# def make_rand_data_rw(mu_min, mu_max, n_samples, g, x, sigma, **kwargs):
#     mu = np.atleast_2d(np.random.rand(n_samples)[:, None] * (mu_max - mu_min) + mu_min)
#     X, X_s = get_data_rw(mu, sigma, g, x, **kwargs)
#     return mu, X, X_s


def L2_error(X, X_truth):
    L2 = np.mean((X - X_truth) ** 2, axis=0) ** 0.5
    return L2


def delta_n_width(S, a, b):
    delta_n = (np.cumsum(S[::-1] ** 2)[::-1] / a / b) ** 0.5
    return delta_n


def optimize_sROMs_naive(case):
    rank = case["rank"]
    # sigma_opt = 1 / (rank * 2)
    sigma_opt = 0.2 / rank  # optimal for saw tooth w=1/45
    case["sigmaD"] = np.zeros((case["n_test"],))

    def target(params):
        case["sigma"], case["sigmaD"][:] = params[0], params[1]
        case["c"] = case["sigmaD"][0]
        return target_function(case)[2]

    # x0 = np.array([sigma_opt, 1.0])
    # res = minimize(
    #     target,
    #     x0,
    #     method="SLSQP",
    #     bounds=[(0.001, 5 * sigma_opt), (0, 10)],
    #     options={"disp": True, "eps": np.array([0.0005, 0.05]), "maxiter": 25, "ftol": 0.0005},
    # )
    bounds = Bounds([0.001, 0.001], [5 * sigma_opt, 5 * sigma_opt])
    res = direct(target, bounds, eps=1e-2, len_tol=0.025)  # max side length_abs=[0.006, 0.25]
    print("optimization result for sigma:", 0.001, res["x"][0], 5 * sigma_opt, sigma_opt)
    print(res.message)
    # TODO: check if bounds were OK
    return res["x"]


def optimize_hyperparameters_3d(case, method="lbfgsb"):
    """
    Optimize (sigma, sigmaD, num_iter) to maximize improvement.
    method: "lbfgsb" (smooth, few evals) or "direct" (robust, more evals).
    Use this instead of DIRECT when the improvement surface is smooth.
    """
    sigma_opt = 0.013
    if "sigmaD" not in case or case["sigmaD"] is None:
        case["sigmaD"] = np.zeros((case["n_test"],))

    def target(params):
        case["sigma"], case["c"], case["num_iter"] = params[0], False, int(round(params[2]))
        case["sigmaD"][:] = params[1]
        return target_function(case)[2]

    bounds = ([0.001, 0.001, 1], [5 * sigma_opt, 5 * sigma_opt, 200])

    if method == "direct":
        res = direct(
            lambda x: -target(x),
            Bounds([b[0] for b in bounds], [b[1] for b in bounds]),
            eps=1e-2,
            len_tol=0.025,
        )
        x_best = res["x"]
        print("DIRECT result:", x_best, "improvement", -res["fun"])
        return x_best

    # L-BFGS-B: minimize -improvement (smooth -> few evaluations)
    def neg_target(x):
        return -target(x)

    # Multi-start to reduce risk of bad local minimum
    starts = [
        np.array([sigma_opt, sigma_opt, 100.0]),
        np.array([0.001, 5 * sigma_opt, 50.0]),
        np.array([5 * sigma_opt, 0.001, 150.0]),
        np.array([sigma_opt * 2, sigma_opt * 0.5, 80.0]),
        np.array([sigma_opt * 0.5, sigma_opt * 2, 120.0]),
    ]
    best_fun = np.inf
    best_x = None
    for x0 in starts:
        res = minimize(
            neg_target,
            x0,
            method="L-BFGS-B",
            bounds=list(zip(bounds[0], bounds[1])),
            options={"maxfun": 80, "ftol": 1e-4},
        )
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x
    x_best = best_x
    print("L-BFGS-B result: sigma=%.5f sigmaD=%.5f num_iter=%d improvement=%.2f" % (x_best[0], x_best[1], int(round(x_best[2])), -best_fun))
    return x_best
