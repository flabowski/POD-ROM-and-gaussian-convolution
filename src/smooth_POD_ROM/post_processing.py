import warnings
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance, cKDTree

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_gpu
except ImportError:
    cp = None
    gaussian_filter_gpu = None


def relative_distance_nearest_neighbour(single_point, points):
    distances = distance.cdist(points, single_point, "euclidean").flatten()
    distances.sort()
    dim = points.shape[1]
    n = 2**dim  # 1D: 2NN; 2D: 4NN, 3D:8NN
    return distances[0] / (np.sum(distances[:n]) / dim)


def get_sigma(sigma, mu, mu_train, c):
    d = relative_distance_nearest_neighbour(mu, mu_train)
    return sigma * (1 + c * d)


def get_sigma_batch(sigma, mu_, mu_train, c):
    tree = cKDTree(mu_train)
    dim = mu_train.shape[1]
    n = 2**dim
    dists, _ = tree.query(mu_, k=n, workers=-1)
    d_ = dists[:, 0] / (dists.sum(axis=1) / dim)
    return sigma * (1 + c * d_)


def _richardson_lucy_truncate(sgm, dx):
    """Truncate for Gaussian kernel (shared by CPU and GPU RL)."""
    truncate = 8
    if sgm / dx > 800:
        truncate = 1
    elif sgm / dx > 400:
        truncate = 2
    elif sgm / dx > 200:
        truncate = 3
    elif sgm / dx > 100:
        truncate = 4
    elif sgm / dx > 50:
        truncate = 5
    elif sgm / dx > 25:
        truncate = 5
    return truncate


def richardson_lucy(
    x,
    im_blur,
    sgm,
    num_iter=50,
    damping=2,
    clip=True,
    mode="wrap",
    monitor_convergence=False,
    clip_min=0,
    clip_max=1,
):
    if not len(im_blur.shape) == 2:
        warnings.warn("image needs to be 2D")
    dx = x[1] - x[0]
    truncate = _richardson_lucy_truncate(sgm, dx)
    im_deconv = im_blur.copy()
    eps = 1e-12  # regularization to avoid 0 division
    if monitor_convergence:
        deconvolved_per_iter = np.empty((im_blur.size, num_iter))
        err = np.empty((num_iter,))
    for k in range(num_iter):
        # blurred = convolve2d(im_deconv.copy(), psf, boundary='symm', mode='same') + eps
        # blurred = convolve_f2D(im_deconv, psf_f2D) + eps
        blurred = gaussian_filter(im_deconv, sigma=sgm / dx, truncate=truncate, mode=mode)  # + eps
        blurred[np.abs(blurred) < eps] = eps
        relative_blur = im_blur / blurred
        # im_deconv *= convolve2d(relative_blur, psf_mirror, boundary='symm', mode='same')
        # im_deconv *= convolve_f2D(relative_blur, psf_f2D)
        error_estimate = gaussian_filter(relative_blur, sigma=sgm / dx, truncate=truncate, mode=mode)
        if damping:
            error_estimate[error_estimate > (1 + damping)] = 1 + damping
            error_estimate[error_estimate < (1 - damping)] = 1 - damping
        im_deconv *= error_estimate
        if clip:
            im_deconv[im_deconv < clip_min] = clip_min
            im_deconv[im_deconv > clip_max] = clip_max
        if monitor_convergence:
            err[k] = np.mean((im_blur - blurred) ** 2) ** 0.5  # np.mean(error_estimate)
            deconvolved_per_iter[:, k] = im_deconv.ravel()
    if monitor_convergence:
        return deconvolved_per_iter, err
    return im_deconv, False


def richardson_lucy_gpu(
    x,
    im_blur,
    sgm,
    num_iter=50,
    damping=2,
    clip=True,
    mode="wrap",
    monitor_convergence=False,
    clip_min=0,
    clip_max=1,
):
    """GPU Richardson–Lucy deconvolution. Requires CuPy. Same signature as richardson_lucy."""
    if cp is None or gaussian_filter_gpu is None:
        raise RuntimeError("richardson_lucy_gpu requires cupy and cupyx.scipy.ndimage")
    if not len(im_blur.shape) == 2:
        warnings.warn("image needs to be 2D")
    if monitor_convergence:
        raise ValueError("use CPU version")
    dx = float(x[1] - x[0])
    truncate = _richardson_lucy_truncate(sgm, dx)

    im_deconv = im_blur.copy()
    eps = 1e-7
    sigma_pixels = sgm / dx
    for k in range(num_iter):
        blurred = gaussian_filter_gpu(im_deconv, sigma=sigma_pixels, truncate=truncate, mode=mode)
        blurred[cp.abs(blurred) < eps] = eps
        relative_blur = im_blur / blurred
        error_estimate = gaussian_filter_gpu(relative_blur, sigma=sigma_pixels, truncate=truncate, mode=mode)
        if damping:
            error_estimate = cp.clip(error_estimate, 1 - damping, 1 + damping)
        im_deconv = im_deconv * error_estimate
        if clip:
            im_deconv = cp.clip(im_deconv, clip_min, clip_max)
    im_deconv_np = cp.asnumpy(im_deconv)
    return im_deconv_np, False


def richardson_lucy2_gpu(
    x,
    im_blur,
    sgm,
    num_iter=50,
    damping=2,
    clip=True,
    mode="wrap",
    clip_min=0,
    clip_max=1,
    filter_epsilon=1e-6,
):
    """GPU Richardson–Lucy deconvolution (improved). Uses filter_epsilon to avoid
    blow-up where blurred is very small: relative_blur = 0 there instead of im_blur/blurred.
    Same signature as richardson_lucy_gpu otherwise."""
    if cp is None or gaussian_filter_gpu is None:
        raise RuntimeError("richardson_lucy2_gpu requires cupy and cupyx.scipy.ndimage")
    if not len(im_blur.shape) == 2:
        warnings.warn("image needs to be 2D")

    dx = float(x[1] - x[0])
    truncate = _richardson_lucy_truncate(sgm, dx)

    im_deconv = im_blur.copy()
    eps = 1e-7
    sigma_pixels = sgm / dx
    for k in range(num_iter):
        blurred = gaussian_filter_gpu(im_deconv, sigma=sigma_pixels, truncate=truncate, mode=mode)
        blurred[cp.abs(blurred) < eps] = eps
        if filter_epsilon is not None and filter_epsilon > 0:
            relative_blur = cp.where(
                blurred >= filter_epsilon,
                im_blur / blurred,
                cp.zeros_like(im_blur),
            )
        else:
            relative_blur = im_blur / blurred
        error_estimate = gaussian_filter_gpu(relative_blur, sigma=sigma_pixels, truncate=truncate, mode=mode)
        if damping:
            error_estimate = cp.clip(error_estimate, 1 - damping, 1 + damping)
        im_deconv = im_deconv * error_estimate
        if clip:
            im_deconv = cp.clip(im_deconv, clip_min, clip_max)
    return im_deconv, False


def post_process(
    x,
    data,
    sigma,
    c,
    mu_test,
    mu_train,
    num_iter,
    shape,
    clip,
    progress=False,
    monitor_convergence=False,
):
    print("post_process:", sigma, c, num_iter, shape, clip)
    # from datetime import datetime
    if monitor_convergence:
        deconvolved = np.ones((*data.shape, num_iter))
    else:
        deconvolved = np.empty_like(data)
    for j in range(len(mu_test)):
        sgm_est = get_sigma(sigma, mu_test[j][None, ...], mu_train, c=c)
        # t1 = datetime.now()
        deconvolved[:, j] = richardson_lucy(
            x,
            data[:, j].reshape(shape),
            sgm_est,
            num_iter,
            # truth=None,
            damping=2,
            clip=clip,
            monitor_convergence=monitor_convergence,
        ).reshape(
            deconvolved[:, j].shape
        )  # n, num_iter
        # print("sgm=", sgm_est*1000, (datetime.now()-t1).total_seconds())
        if progress:
            print(j, end=", ")
    return deconvolved


def sharpen(data, s, ns):
    # assert s % 2 == 1, "must be odd"
    data_sharpened = data.copy()
    if s == 1:
        return data.ravel()
    elif s == 3:
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif s == 5:
        k = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, -1, 0, 0],
                [-1, -1, 9, -1, -1],
                [0, 0, -1, 0, 0],
                [0, 0, -1, 0, 0],
            ]
        )
    elif s == 7:
        k = np.array(
            [
                [0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0, 0],
                [-1, -1, -1, 13, -1, -1, -1],
                [0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0, 0],
            ]
        )
    elif s == 9:
        k = np.array(
            [
                [0, 0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0],
                [-1, -1, -1, -1, 17, -1, -1, -1, -1],
                [0, 0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0],
            ]
        )
    elif s == 11:
        k = np.array(
            [
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1, 21, -1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            ]
        )
    else:
        raise ValueError("unknown size s ={:.0f}".format(s))
    for i in range(ns):
        data_sharpened = convolve2d(data_sharpened, k, boundary="symm", mode="same")
    return data_sharpened.ravel()


def remove_wall(data, X, Y):
    data = data.copy()
    # [2, 0, 0],
    # [2, 0.32876, 0],
    # [2.16438, 0.32876, 0],
    # [2.16438, 0, 0],
    is_wall = (0.292 <= X) & (X <= 0.316) & (Y <= 0.048)
    # data[47:54, 0:16] = .5
    data[is_wall] = -1
    return data


def threshold(data):
    data = data.copy()
    data[data < 0.5] = 0
    data[data >= 0.5] = 1
    return data
